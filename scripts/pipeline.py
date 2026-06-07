#!/usr/bin/env python3
"""
Chexie Knowledge Pipeline
Enhanced search: full-thread retrieval, BM25+vector hybrid,
multi-query expansion, RRF fusion, content-filtered output.

Usage:
    python3 pipeline.py "25双日白河A组出现了什么问题"
    python3 pipeline.py --interactive
    python3 pipeline.py --top-k 30 "白河 集合 迟到"
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "faiss_index.bin"
META_PATH = ROOT / "faiss_meta.jsonl"
BM25_CACHE = ROOT / "bm25_index.jsonl"
MODEL_NAME = "shibing624/text2vec-base-chinese"

_meta_cache = None
_thread_cache = None
_faiss_cache = None
_bm25_cache = None

LOCATION_ALIASES = {
    "白河": "白河", "官厅": "官厅", "松山": "松山", "琉璃庙": "琉璃庙",
    "南石洋": "南石洋", "十渡": "十渡", "禅房": "禅房", "潭柘寺": "潭柘寺",
    "妙峰": "妙峰", "黄花城": "黄花城", "八达岭": "八达岭", "白羊沟": "白羊沟",
    "蟒山": "蟒山", "慕田峪": "慕田峪", "潭柘寺": "潭柘寺",
}

# ── Content filter keywords ──────────────────────────────────────

# Keep chunks containing these (problem + solution content)
KEEP_PATTERNS = [
    r'问题与思考', r'问题与总结', r'情况说明', r'问题与讨论',
    r'解决方案', r'个人思考', r'反思[：:]', r'总结与反思',
    r'需要执委会讨论的问题', r'出现的问题', r'存在问题',
    r'第\s*[一二三四五六七八九十]+\s*[．、]',  # "三．" section headers
    r'工作记录', r'时间线', r'人员组成', r'职务安排',
    r'路线情况', r'注意事项', r'物资准备',
    r'拉练概况', r'集合时间', r'出发时间',
    r'检车安排', r'押后负责', r'队医负责',
]

# Skip chunks that are purely personal praise
SKIP_PATTERNS = [
    r'感谢.*辛苦啦', r'夸！', r'夸～', r'qwq',
    r'期待下次.*一起骑车', r'希望.*在协会玩得开心',
    r'希望.*一切顺利', r'希望.*技术进步',
    r'超级棒', r'超级靠谱', r'超级厉害',
    r'加油', r'祝.*顺利', r'最棒啦', r'大家好好',
    r'终于当上了', r'滑跪道歉',
]

# ── Data Loaders ─────────────────────────────────────────────────

def load_meta() -> list[dict]:
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    rows = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    _meta_cache = rows
    return rows

def get_thread_index() -> dict[str, list[dict]]:
    global _thread_cache
    if _thread_cache is not None:
        return _thread_cache
    meta = load_meta()
    threads = defaultdict(list)
    for item in meta:
        tid = item.get("meta", {}).get("tid", "")
        if tid:
            threads[tid].append(item)
    for tid in threads:
        threads[tid].sort(key=lambda x: x.get("meta", {}).get("chunk_index", 0))
    _thread_cache = dict(threads)
    return _thread_cache

def get_faiss():
    global _faiss_cache
    if _faiss_cache is not None:
        return _faiss_cache
    index = faiss.read_index(str(INDEX_PATH))
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    _faiss_cache = (index, model)
    return _faiss_cache

def get_bm25():
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache
    if not BM25_CACHE.exists():
        return None
    texts, chunk_ids, tokenized = [], [], []
    with BM25_CACHE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                tokenized.append(entry["tokens"])
                texts.append(entry["text"])
                chunk_ids.append(entry["id"])
    bm25 = BM25Okapi(tokenized)
    _bm25_cache = (bm25, texts, chunk_ids)
    return _bm25_cache

# ── Text Cleanup ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'【\s*第\d+楼.*?】', '', text)
    text = re.sub(r'(?<=\d)\s*\n\s*(?=\d)', '', text)
    text = re.sub(r'(?<=\d)\s*\n\s*(?=[\d.])', '', text)
    text = re.sub(r'(?<![。；，、）】」\n])\n(?![。；，、）】」\n])', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text

def is_valuable_chunk(text: str) -> bool:
    """Check if a chunk contains problem-solving content worth keeping."""
    if not text or len(text) < 30:
        return False
    # Skip pure praise/thanks
    skip_score = 0
    for pat in SKIP_PATTERNS:
        if re.search(pat, text):
            skip_score += 1
    if skip_score >= 2:
        return False
    # Check for keep patterns
    for pat in KEEP_PATTERNS:
        if re.search(pat, text):
            return True
    # Also keep if it has substantial problem-related content
    problem_words = ["问题", "注意", "建议", "可以", "需要", "应该",
                     "迟到", "摔车", "扎胎", "夜路", "放坡", "爬坡",
                     "留口", "前旗", "后旗", "前站", "押后", "队医",
                     "绑包", "拖沓", "职务", "路线"]
    score = sum(1 for w in problem_words if w in text)
    if score >= 3 and len(text) > 100:
        return True
    # Always keep metadata chunks
    meta_words = ["人员", "里程", "爬升", "坡", "路线", "路书", "集合"]
    if any(w in text[:200] for w in meta_words) and len(text) > 80:
        return True
    return False

def filter_thread_chunks(chunks: list[dict]) -> list[dict]:
    """Keep only valuable chunks from a thread, filter out praise/thanks."""
    filtered = []
    for c in chunks:
        text = c.get("text", "")
        if is_valuable_chunk(text):
            filtered.append(c)
    # If filtering removed everything, return original (safety net)
    if not filtered and chunks:
        return chunks[:2]
    return filtered

def thread_info(chunks: list[dict]) -> dict:
    if not chunks:
        return {}
    m = chunks[0].get("meta", {})
    return {
        "tid": m.get("tid", ""),
        "title": m.get("title", ""),
        "author": m.get("thread_author", ""),
        "date": m.get("thread_date", ""),
        "bid": m.get("bid", ""),
        "board": m.get("board", ""),
        "url": m.get("url", "").split("&p=")[0] if m.get("url") else "",
        "source_label": m.get("source_label", ""),
    }

# ── Year/Group utils ─────────────────────────────────────────────

def years_match(parsed_year: str | None, title: str) -> bool:
    if not parsed_year:
        return True
    if parsed_year in title:
        return True
    if len(parsed_year) == 4:
        short = parsed_year[2:]
        if short in title:
            return True
    return False

# ── Search ───────────────────────────────────────────────────────

def faiss_search(query: str, top_k: int = 25) -> list[dict]:
    index, model = get_faiss()
    meta = load_meta()
    emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(emb, dtype=np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        results.append({
            "id": meta[idx]["id"], "text": meta[idx]["text"],
            "meta": meta[idx].get("meta", {}), "score": float(score),
        })
    return results

def bm25_search(query: str, top_k: int = 30) -> list[dict]:
    bm25_data = get_bm25()
    if bm25_data is None:
        return []
    bm25, texts, chunk_ids = bm25_data
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    top = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top:
        if scores[idx] <= 0:
            continue
        results.append({"id": chunk_ids[idx], "text": texts[idx],
                        "score": float(scores[idx])})
    return results

# ── Query ────────────────────────────────────────────────────────

def parse_query(query: str) -> dict:
    r = {"year": None, "group": None, "group_type": None, "group_num": None,
         "location": None, "is_problem_query": False, "topics": []}
    ym = re.search(r'(20[2-9]\d|(?<!\d)([1-2]\d)(?!\d))', query)
    if ym:
        y = ym.group(1)
        r["year"] = "20" + y if len(y) == 2 else y
    gm = re.search(r'([AB])\s*(\d)?\s*(?:组)?', query)
    if gm:
        r["group_type"] = gm.group(1)
        if gm.group(2):
            r["group_num"] = gm.group(2)
            r["group"] = f"{gm.group(1)}{gm.group(2)}"
        else:
            r["group"] = f"{gm.group(1)}组"
    for alias, loc in LOCATION_ALIASES.items():
        if alias in query:
            r["location"] = loc
            break
    for kw in ["问题", "事故", "摔车", "检讨", "错误", "不足", "注意",
                "教训", "经验", "总结", "建议", "改进", "复盘", "怎么办"]:
        if kw in query:
            r["is_problem_query"] = True
            break
    return r

def expand_queries(parsed: dict) -> list[str]:
    base = parsed.get("raw", "")
    expanded = [base]
    year, loc = parsed.get("year",""), parsed.get("location","")
    prefix = ""
    if year and loc:
        prefix = f"{year}双日{loc}"
    if parsed["is_problem_query"] and prefix:
        expanded.append(f"{prefix} 队长总结")
        expanded.append(f"{prefix} 问题 思考")
    elif parsed["is_problem_query"]:
        expanded.append("队长总结 问题与思考")
    seen = set()
    return [q for q in (x.strip() for x in expanded) if q and q not in seen and not seen.add(q)]

# ── RRF ──────────────────────────────────────────────────────────

def rrf_fuse(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    scores, items = defaultdict(float), {}
    for rlist in results_list:
        for rank, item in enumerate(rlist):
            cid = item["id"]
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in items:
                items[cid] = item
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for cid, score in ordered:
        items[cid]["rrf_score"] = round(score, 6)
    return [items[cid] for cid, _ in ordered]

# ── Target Detection ─────────────────────────────────────────────

def find_target_thread(parsed: dict) -> dict | None:
    year, loc, group = parsed.get("year"), parsed.get("location"), parsed.get("group")
    if not year and not loc:
        return None
    threads = get_thread_index()
    candidates = []
    for tid, chunks in threads.items():
        info = thread_info(chunks)
        title = info.get("title", "")
        score = 0
        if years_match(year, title):
            score += 3
        if loc and loc in title:
            score += 3
        if group:
            if group in title:
                score += 2
            elif group[0] in title:
                score += 1
        if "队长总结" in title:
            score += 2
        if year and loc and f"{year}双日{loc}" in title.replace(" ", ""):
            score += 3
            if group and group[0] in title:
                score += 2
        if score >= 5:
            candidates.append((score, tid, info, chunks))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, tid, info, chunks = candidates[0]
    return {"tid": tid, "info": info, "chunks": chunks, "match_score": score}

# ── Thread Aggregation ───────────────────────────────────────────

def aggregate_by_thread(chunks: list[dict]) -> list[dict]:
    threads = get_thread_index()
    t_scores, t_chunks = defaultdict(float), defaultdict(list)
    for item in chunks:
        tid = item.get("meta", {}).get("tid", "")
        if not tid:
            continue
        s = item.get("rrf_score", item.get("score", 0))
        t_scores[tid] = max(t_scores[tid], s)
        t_chunks[tid].append(item)
    results = []
    for tid, score in sorted(t_scores.items(), key=lambda x: x[1], reverse=True):
        all_c = threads.get(tid, t_chunks[tid])
        filtered = filter_thread_chunks(all_c)
        results.append({
            "tid": tid, "info": thread_info(all_c),
            "score": round(score, 4),
            "filtered_chunks": filtered,
            "total_chunks": len(all_c),
        })
    return results

# ── Output ───────────────────────────────────────────────────────

def extract_problems(chunks: list[dict]) -> list[str]:
    """Extract problem+solution items, deduplicating chunk boundary overlap."""
    # Find the chunk that contains "问题与思考" or similar
    main_text = ""
    found = False
    for c in chunks:
        text = c.get("text", "")
        if not found and ("问题与思考" in text or "问题与总结" in text or "问题与讨论" in text):
            # Start from the section marker
            for marker in ["问题与思考", "问题与总结", "问题与讨论"]:
                idx = text.find(marker)
                if idx >= 0:
                    text = text[idx:]
                    break
            found = True
        if found:
            # Skip pure photo/thanks sections
            if any(k in text[:80] for k in ["七：", "八：", "想说的话", "致谢", "一些照片"]):
                break
            clean = clean_text(text)
            if len(clean) > 30:
                main_text += clean + "\n"

    if not main_text:
        return []

    # Deduplicate: split at numbered items and remove near-duplicates
    items = re.split(r'(?=\d+\.)', main_text)
    seen = set()
    unique_items = []
    for item in items:
        item = item.strip()
        if not item or len(item) < 20:
            continue
        # Extract just the problem number + first 60 chars for dedup
        m = re.match(r'(\d+\.)\s*(.*?)(?:\n|$)', item)
        key = m.group(0)[:80] if m else item[:80]
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    # If first item is a fragment (no number), drop it
    if unique_items and not re.match(r'\d+\.', unique_items[0]):
        unique_items = unique_items[1:]

    return unique_items[:10]

def build_output(target, threads, parsed, siblings=None) -> str:
    lines, sep = [], "=" * 60
    year, loc = parsed.get("year",""), parsed.get("location","")

    # ═══ Layer 1: Target ═══
    if target:
        info = target["info"]
        lines.append(sep)
        lines.append(f"【目标拉练】{info['title']}")
        lines.append(f"  作者：{info.get('author','')} | 日期：{info.get('date','')}")
        lines.append(f"  链接：{info.get('url','')}")
        lines.append("")

        # Show problems + solutions + reflections
        problems = extract_problems(target["chunks"])
        if problems:
            lines.append("── 问题与解决方案 ──")
            for p in problems[:7]:
                lines.append("")
                lines.append(p)
            lines.append("")
        else:
            # Fallback: show filtered chunks directly
            filtered = filter_thread_chunks(target["chunks"])
            for c in filtered[:5]:
                text = clean_text(c.get("text", ""))
                if text:
                    lines.append(clean_text(text[:2000]))
                    lines.append("")

    # ═══ Layer 2: Siblings ═══
    sibling = siblings or []
    if not sibling:
        for tr in threads[:15]:
            if target and tr["tid"] == target["tid"]:
                continue
            t = tr["info"].get("title","")
            if years_match(year, t) and loc and loc in t:
                sibling.append(tr)

    if sibling:
        lines.append(sep)
        lines.append(f"【同期对照·{year or ''}{loc or ''}】")
        for i, tr in enumerate(sibling[:5]):
            info = tr["info"]
            lines.append(f"\n  [{i+1}] {info['title']}")
            lines.append(f"      作者：{info.get('author','')} | {info.get('date','')}")
            lines.append(f"      链接：{info.get('url','')}")
            # Show a brief summary of problems if available
            probs = extract_problems(tr.get("filtered_chunks", tr.get("chunks", [])))
            if probs:
                summary = clean_text(probs[0])
                if len(summary) > 300:
                    summary = summary[:300] + "..."
                lines.append(f"      {summary}")
            else:
                # Show first filtered chunk if no problems extracted
                chunks = tr.get("filtered_chunks", tr.get("chunks", []))
                for c in chunks[:1]:
                    ct = clean_text(c.get("text", ""))
                    if ct:
                        lines.append(f"      {ct[:200]}...")
                        break
        lines.append("")

    # ═══ Layer 3: Historical ═══
    historical = []
    for tr in threads[:15]:
        if target and tr["tid"] == target["tid"]:
            continue
        if any(s["tid"] == tr["tid"] for s in sibling):
            continue
        t = tr["info"].get("title","")
        if loc and loc in t and "队长总结" in t:
            historical.append(tr)
    if historical:
        lines.append(sep)
        lines.append(f"【往年经验·{loc or ''}】")
        for i, tr in enumerate(historical[:6]):
            info = tr["info"]
            lines.append(f"\n  [{i+1}] {info['title']}")
            lines.append(f"      作者：{info.get('author','')} | {info.get('date','')}")
            lines.append(f"      链接：{info.get('url','')}")
            problems = extract_problems(tr.get("filtered_chunks", tr.get("chunks", [])))
            if problems:
                summary = clean_text(problems[0])
                if len(summary) > 400:
                    summary = summary[:400] + "..."
                lines.append(f"      {summary}")
        lines.append("")

    return "\n".join(lines)

# ── Pipeline ─────────────────────────────────────────────────────

def search_pipeline(query: str, top_k: int = 20, use_bm25: bool = False,
                    verbose: bool = False) -> dict:
    t0 = time.time()
    result = {"query": query, "target": None, "threads": [], "output": ""}

    parsed = parse_query(query)
    parsed["raw"] = query
    result["parsed"] = parsed
    if verbose:
        y, g, l = parsed["year"], parsed["group"], parsed["location"]
        print(f"[parse] year={y} group={g} loc={l} problem={parsed['is_problem_query']}",
              flush=True)

    target = find_target_thread(parsed)
    result["target"] = target
    if verbose and target:
        ttl = target["info"].get("title", "")
        print(f"[target] {ttl} (tid={target['tid']})", flush=True)

    queries = expand_queries(parsed)
    if verbose:
        print(f"[expand] {len(queries)} queries", flush=True)

    t1 = time.time()
    all_faiss = [faiss_search(q, top_k) for q in queries]
    if verbose:
        print(f"[faiss] {time.time()-t1:.1f}s", flush=True)

    t1 = time.time()
    bm25_res = []
    if use_bm25:
        bm25_res = bm25_search(query, top_k * 2)
    if verbose:
        print(f"[bm25] {time.time()-t1:.1f}s ({len(bm25_res)} results)", flush=True)

    sources = all_faiss + ([bm25_res] if bm25_res else [])
    fused = rrf_fuse(sources, k=60)
    if verbose:
        print(f"[rrf] {len(fused)} unique chunks", flush=True)

    meta = load_meta()
    by_id = {m["id"]: m for m in meta}
    for item in fused:
        if "meta" not in item and item["id"] in by_id:
            item["meta"] = by_id[item["id"]].get("meta", {})

    threads = aggregate_by_thread(fused)
    result["threads"] = threads
    if verbose:
        print(f"[aggregate] {len(threads)} threads", flush=True)

    # Siblings via index scan
    sibling_threads = []
    if parsed.get("year") and parsed.get("location"):
        year_short = parsed["year"][2:] if len(parsed["year"]) == 4 else parsed["year"]
        loc = parsed["location"]
        candidates = []
        for tid, chunks in get_thread_index().items():
            info = thread_info(chunks)
            title = info.get("title", "")
            if (year_short in title and loc in title and "双日" in title
                    and ("队长总结" in title or "拉练通知" in title)):
                if not (target and tid == target["tid"]):
                    candidates.append((tid, title))
        candidates.sort(key=lambda x: (2 if "队长总结" in x[1] else 1), reverse=True)
        for tid, title in candidates[:6]:
            all_c = get_thread_index().get(tid, [])
            if all_c:
                chunk_kept = filter_thread_chunks(all_c)
                sibling_threads.append({
                    "tid": tid, "info": thread_info(all_c),
                    "score": 1.0, "filtered_chunks": chunk_kept, "total_chunks": len(all_c),
                })
    if verbose:
        print(f"[sibling] {len(sibling_threads)} threads", flush=True)

    result["output"] = build_output(target, threads, parsed, siblings=sibling_threads)
    result["time"] = round(time.time() - t0, 2)
    if verbose:
        print(f"[total] {result['time']}s", flush=True)

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chexie Knowledge Pipeline")
    ap.add_argument("query", nargs="?", help="Search query")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--interactive", "-i", action="store_true")
    args = ap.parse_args()

    if args.interactive or not args.query:
        print("Chexie Pipeline (q to quit)")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("q", "quit", "exit"):
                break
            if not q:
                continue
            r = search_pipeline(q, top_k=args.top_k, verbose=args.verbose)
            print(r["output"])
            print(f"\n  [{r['time']}s]")
    else:
        r = search_pipeline(args.query, top_k=args.top_k, verbose=args.verbose)
        print(r["output"])
        print(f"\n  [{r['time']}s]")
