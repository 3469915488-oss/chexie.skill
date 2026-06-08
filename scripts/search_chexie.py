#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path
import sqlite3
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(os.environ.get("CHEXIE_ROOT", "/opt/chexie-knowledge"))
INDEX_PATH = ROOT / "faiss_index.bin"
META_PATH = ROOT / "faiss_meta.jsonl"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_DEFAULT_MODEL_DIR = os.environ.get("CHEXIE_MODEL_DIR", "/opt/wiki/models")
MODEL_PATH = Path(_DEFAULT_MODEL_DIR) / (
    "models--BAAI--bge-small-zh-v1.5/"
    "snapshots/7999e1d3359715c523056ef9478215996d62a620"
)
FTS5_DB_PATH = ROOT / "chexie_fts.db"

# Known author IDs for FTS5 entity search
KNOWN_AUTHORS = [
    'dudu', '劳模', '温瑶', '小仓鼠', '蓝', '牛肉汤圆', '栖风',
    '秋小果', '问道', '月霜', '踏月', '小瓜', '华年', '森云',
    '小笨熊', '游乐', '六三', '飞舟', '河虾', '橘猫', '上线',
    '伯约', '沐阳', '烟火', '碧瞳', '壹乙', '德克士',
]

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# === Board weights for authority-aware ranking ===
BOARD_WEIGHTS = {
    "车协工作区": 4.5,
    "行者足音": 3.0,
    "一技之长": 2.0,
    "车友宝典": 1.5,
    "纯净水": 1.0,
}
SOURCE_LABEL_WEIGHT = {"执委会": 0.5, "理事会": 0.5, "通知": 0.3, "制度": 0.3, "总结": 0.3}

# === Entity index paths ===
ENTITY_DIR = ROOT / "entities"
TITLE_INDEX_PATH = ROOT / "title_index.bin"
TITLE_META_PATH = ROOT / "title_meta.jsonl"

# === Query type patterns ===
EXACT_PATTERNS = [
    r'(?:白河|禅房|十渡|妙峰|潭柘寺|九龙山|白羊沟|凤凰岭|慕田峪|十三陵|香山)',
    r'(?:202[0-9]|19[0-9]{2}|[0-9]{2}(?:春|秋|暑|冬))',
    r'(?:执委会|理事会|通知|制度|财务|报销|体测|报名)',
]
FUZZY_PATTERNS = [
    r'(?:问题|怎么办|处理|解决|方案|争议|讨论|怎么看|经验|教训|总结|复盘)',
    r'(?:出了|遇到过|以前|之前|之前有没有|历史)',
    r'(?:请教|求助|求问|想问|问一下)',
]

KEYWORDS = [
    "远征",
    "报名",
    "财务",
    "预算",
    "报销",
    "执委会",
    "理事会",
    "拉练",
    "体测",
    "装备",
    "团购",
    "十渡",
    "妙峰",
    "春训",
    "双日",
    "单日",
    "保险",
    "路线",
    "路书",
    "队医",
    "外联",
]

EVENT_TERMS = ["十渡", "妙峰", "八大处", "戒台寺", "东方红", "卢沟桥", "十三陵", "香山", "凤凰岭"]

CUE_KEYWORDS = {
    "proposal": ("提案", "建议", "方案", "讨论", "是否", "可以", "希望"),
    "practice": ("做法", "流程", "规定", "安排", "通知", "执行", "总结"),
    "benefit": ("好处", "优点", "优势", "方便", "提高", "减少"),
    "risk": ("坏处", "问题", "风险", "争议", "不足", "担心", "缺点"),
    "outcome": ("结果", "最后", "通过", "决定", "落实", "复盘"),
}


def load_meta(path: Path = META_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def resources() -> tuple[Any, list[dict[str, Any]], SentenceTransformer]:
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_meta()
    model_ref = str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_NAME
    model = SentenceTransformer(model_ref, device="cpu")
    return index, meta, model


@lru_cache(maxsize=1)
def title_resources() -> tuple[Any, list[dict[str, Any]], SentenceTransformer] | None:
    """Load title index + meta. Returns None if not yet built."""
    if not TITLE_INDEX_PATH.exists() or not TITLE_META_PATH.exists():
        return None
    t_idx = faiss.read_index(str(TITLE_INDEX_PATH))
    t_meta: list[dict[str, Any]] = []
    with TITLE_META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t_meta.append(json.loads(line))
    _, _, model = resources()  # reuse same model
    return t_idx, t_meta, model


@lru_cache(maxsize=1)
def entity_index() -> dict[str, dict[str, list[int]]]:
    entities: dict[str, dict[str, list[int]]] = {}
    for name in ("routes", "problems", "roles", "years"):
        path = ENTITY_DIR / f"{name}.json"
        if path.exists():
            entities[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            entities[name] = {}
    return entities


def source_of(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source") or row.get("meta") or {}
    if not isinstance(source, dict):
        source = {}
    return source


def detect_query_type(query: str) -> str:
    """Classify query as 'exact', 'fuzzy', or 'mixed'."""
    import re
    exact_score = 0
    fuzzy_score = 0
    for pat in EXACT_PATTERNS:
        if re.search(pat, query):
            exact_score += 1
    for pat in FUZZY_PATTERNS:
        if re.search(pat, query):
            fuzzy_score += 1
    if exact_score > 0 and fuzzy_score > 0:
        return "mixed"
    if exact_score > 0:
        return "exact"
    if fuzzy_score > 0:
        return "fuzzy"
    return "mixed"  # default


def board_weight(source: dict[str, Any]) -> float:
    board = str(source.get("board") or "")
    bw = BOARD_WEIGHTS.get(board, 1.0)
    label = str(source.get("source_label") or "")
    for kw, lw in SOURCE_LABEL_WEIGHT.items():
        if kw in label or kw in str(source.get("title") or ""):
            bw += lw
    return bw


def text_of(row: dict[str, Any]) -> str:
    return str(row.get("text") or "")


def keyword_boost(query: str, row: dict[str, Any]) -> float:
    source = source_of(row)
    board = str(source.get("board") or "")
    title = str(source.get("title") or "")
    label = str(source.get("source_label") or "")
    source_type = str(source.get("source_type") or "")
    text = text_of(row)
    haystack = " ".join(
        str(x or "")
        for x in [
            board,
            title,
            label,
            source_type,
            " ".join(source.get("cues") or []),
            text,
        ]
    )
    boost = 0.0
    for word in KEYWORDS:
        if word not in query:
            continue
        if word in title:
            boost += 0.09
        elif word in label:
            boost += 0.075
        elif word in text:
            boost += 0.035
        elif word in haystack:
            boost += 0.02
    if ("执委会" in query or "理事会" in query) and (
        "执委会" in title or "理事会" in title or "执委会" in label or "理事会" in label
    ):
        boost += 0.12
    if any(word in query for word in ("通知", "制度", "总结", "报名", "财务")) and board == "车协工作区":
        boost += 0.04
    if any(word in query for word in ("争议", "建议", "提案", "讨论")) and board in {"行者足音", "纯净水"}:
        boost += 0.025
    cues = set(source.get("cues") or [])
    for cue, words in CUE_KEYWORDS.items():
        if cue in cues and any(word in query for word in words):
            boost += 0.035
    if board in {"车协工作区", "行者足音", "纯净水"}:
        boost += 0.01
    return min(boost, 0.45)


def fts5_search(
    keywords: list[str] | None = None,
    author: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    board: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Search chexie_fts.db using SQL LIKE for robust keyword/author matching.

    Solves the problem where FTS5 MATCH fails on English tokens (e.g. 'dudu')
    and FAISS semantic search cannot do entity+time-window cross-location.
    """
    if not FTS5_DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(FTS5_DB_PATH))
    cursor = conn.cursor()

    conditions: list[str] = []
    params: list = []

    if author:
        conditions.append('author = ?')
        params.append(author)
    if board:
        conditions.append('board = ?')
        params.append(board)
    if keywords:
        kw_parts = []
        for kw in keywords:
            kw_parts.append('text LIKE ?')
            params.append(f'%{kw}%')
        conditions.append('(' + ' OR '.join(kw_parts) + ')')

    if time_start:
        # Use SQL LIKE to narrow to approximate time range
        # If range spans multiple years, use just the year prefix
        if time_end and time_start[:4] != time_end[:4]:
            # Multi-year range: match either year
            ys = time_start[:4]
            ye = time_end[:4]
            conditions.append('(text LIKE ? OR text LIKE ?)')
            params.extend([f'%{ys}%', f'%{ye}%'])
        else:
            ym = time_start[:7]  # e.g. "2025-05"
            conditions.append('text LIKE ?')
            params.append(f'%{ym}%')
    
    where = ' AND '.join(conditions) if conditions else '1=1'
    query_sql = f"SELECT chunk_id, title, board, author, text FROM fts5_index WHERE {where} LIMIT ?"
    params.append(limit)

    try:
        cursor.execute(query_sql, params)
        rows = cursor.fetchall()
    except Exception:
        conn.close()
        return []
    conn.close()

    results: list[dict[str, Any]] = []
    for row in rows:
        chunk_id, title, board_val, author_val, text = row

        # Post-filter by time range
        times = re.findall(r'20\d{2}-\d{2}-\d{2}', text[:500])
        if time_start and times and max(times) < time_start:
            continue
        if time_end and times and min(times) > time_end:
            continue

        # Parse source from chunk_id
        bid, tid = 1, 0
        for p in chunk_id.split('_'):
            if p.startswith('bid'):
                try: bid = int(p[3:])
                except: pass
            elif p.startswith('tid'):
                try: tid = int(p[3:])
                except: pass

        time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text[:500])
        post_time = time_match.group(1) if time_match else ''
        floor_match = re.search(r'第(\d+)楼', text[:500])
        floor = floor_match.group(1) if floor_match else '?'

        source = {
            'bid': bid, 'tid': tid, 'board': board_val, 'author': author_val,
            'title': title, 'floor': floor, 'post_time': post_time,
            'url': f'https://www.chexie.net/bbs/content/index.php?bid={bid}&tid={tid}&p=1#第{floor}楼',
        }
        results.append({
            'chunk_id': chunk_id, 'text': text, 'source': source,
            'score': 0.0, 'rerank_score': 0.55,
        })

    return results


def _fts5_supplement(query: str, existing: list[dict], top_k: int) -> list[dict[str, Any]]:
    """Auto-extract keywords and authors from query, run FTS5 LIKE search."""
    author_hints = [a for a in KNOWN_AUTHORS if a in query]
    EVENT_HINTS = ['分团', '两团', '三团', '挂人', '团长', '理事会',
                   '暑期', '春训', '互评', '陇青', '命运', '应带尽带',
                   '温瑶', '小仓鼠', '牛肉汤圆', '追究', '处罚']
    kw_hints = [kw for kw in EVENT_HINTS if kw in query]
    # If query mentions specific people, add them as keywords too
    # But skip if that person is already being used as an author filter
    for a in KNOWN_AUTHORS:
        if a in query and a not in kw_hints and a not in author_hints:
            kw_hints.append(a)
    if not author_hints and not kw_hints:
        return []

    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()
    is_25_event = '25' in query or '分团' in query
    t_start = '2025-05-01' if is_25_event else None
    t_end = '2026-06-30' if is_25_event else None

    for author in (author_hints or [None]):
        fts_results = fts5_search(
            keywords=kw_hints if kw_hints else None,
            author=author, time_start=t_start, time_end=t_end, limit=30,
        )
        for r in fts_results:
            if r['chunk_id'] not in seen:
                seen.add(r['chunk_id'])
                all_results.append(r)

    return all_results[:top_k]


def _thread_expand(query: str, existing: list[dict], top_k: int) -> list[dict[str, Any]]:
    """For threads already found by FAISS, pull posts by other key authors in the same thread.
    
    Example: FAISS finds 劳模's post in tid=9015. _thread_expand then searches for
    温瑶, 问道, 栖风 etc. in tid=9015, even if their posts don't contain the query keywords.
    """
    if not FTS5_DB_PATH.exists():
        return []
    
    # Collect tids from FAISS results
    faiss_tids: set[int] = set()
    for item in existing[:15]:  # Only expand from top FAISS results
        src = item.get('source', {})
        tid = src.get('tid')
        if tid and isinstance(tid, int) and tid > 0:
            faiss_tids.add(tid)
    
    if not faiss_tids:
        return []
    
    # Find authors mentioned in query
    query_authors = [a for a in KNOWN_AUTHORS if a in query]
    if not query_authors:
        return []
    
    conn = sqlite3.connect(str(FTS5_DB_PATH))
    cursor = conn.cursor()
    
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    
    for tid in faiss_tids:
        tid_pattern = f'_tid{tid}_'
        for author in query_authors:
            try:
                cursor.execute(
                    "SELECT chunk_id, title, board, author, text FROM fts5_index "
                    "WHERE chunk_id LIKE ? AND author = ? LIMIT 10",
                    (f'%{tid_pattern}%', author)
                )
                rows = cursor.fetchall()
            except Exception:
                continue
            
            for row in rows:
                chunk_id, title, board_val, author_val, text = row
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text[:500])
                post_time = time_match.group(1) if time_match else ''
                floor_match = re.search(r'第(\d+)楼', text[:500])
                floor = floor_match.group(1) if floor_match else '?'
                
                source = {
                    'bid': 1, 'tid': tid, 'board': board_val, 'author': author_val,
                    'title': title, 'floor': floor, 'post_time': post_time,
                    'url': f'https://www.chexie.net/bbs/content/index.php?bid=1&tid={tid}&p=1#第{floor}楼',
                }
                results.append({
                    'chunk_id': chunk_id, 'text': text, 'source': source,
                    'score': 0.0, 'rerank_score': 0.58,  # Slightly above FTS5 base
                })
    
    conn.close()
    return results[:top_k]


def search(query: str, top_k: int, candidate_k: int | None = None) -> dict[str, Any]:
    index, meta, model = resources()
    if candidate_k is None:
        candidate_k = max(top_k * 30, 150)
    candidate_k = min(candidate_k, len(meta))

    vec = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True, show_progress_bar=False)
    vec = np.asarray(vec, dtype="float32")
    scores, ids = index.search(vec, candidate_k)

    ranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    seen_sources: set[tuple[str, str, str, str]] = set()
    for score, idx in zip(scores[0], ids[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(meta) or idx in seen:
            continue
        seen.add(idx)
        row = meta[idx]
        source = source_of(row)
        source["url"] = fix_url(source)
        source_key = (
            str(source.get("board") or ""),
            str(source.get("tid") or source.get("title") or ""),
            str(source.get("floor") or ""),
        )
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        boost = keyword_boost(query, row)
        ranked.append(
            {
                "id": row.get("id") or idx,
                "score": float(score),
                "rerank_score": float(score) + boost,
                "text": text_of(row),
                "source": source,
            }
        )

    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    ranked = merge_exact_event_hits(query, ranked, meta, top_k)

    # ── FTS5 LIKE merge: supplement results that FAISS misses ──
    fts5_results = _fts5_supplement(query, ranked, top_k)
    # Also do thread-based expansion: pull other authors from FAISS-found threads
    thread_results = _thread_expand(query, ranked, top_k)
    all_supplements = fts5_results + thread_results
    if all_supplements:
        existing_keys = {
            (str(item['source'].get('board') or ''),
             str(item['source'].get('tid') or ''),
             str(item['source'].get('floor') or ''))
            for item in ranked
        }
        for fts_item in all_supplements:
            key = (str(fts_item['source'].get('board') or ''),
                   str(fts_item['source'].get('tid') or ''),
                   str(fts_item['source'].get('floor') or ''))
            if key not in existing_keys:
                ranked.append(fts_item)
                existing_keys.add(key)
        ranked.sort(key=lambda item: item['rerank_score'], reverse=True)

    return {
        "question": query,
        "model": MODEL_NAME,
        "index": str(INDEX_PATH),
        "count": len(meta),
        "results": ranked[:top_k],
    }


def merge_exact_event_hits(
    query: str,
    ranked: list[dict[str, Any]],
    meta: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    event_terms = [term for term in EVENT_TERMS if term in query]
    if not event_terms or not any(word in query for word in ("报名", "人数", "名额", "太多", "满", "截止")):
        return ranked

    existing_keys = {
        (
            str(item["source"].get("board") or ""),
            str(item["source"].get("tid") or item["source"].get("title") or ""),
            str(item["source"].get("floor") or ""),
        )
        for item in ranked
    }
    exact: list[dict[str, Any]] = []
    for idx, row in enumerate(meta):
        source = source_of(row)
        title = str(source.get("title") or "")
        text = text_of(row)
        haystack = f"{title}\n{text}"
        if not all(term in haystack for term in event_terms):
            continue
        has_handling_cue = any(word in text for word in ("人数过多", "报名人数", "名额", "过多", "太多", "截止报名", "协商调整", "调剂"))
        has_event_title = all(term in title for term in event_terms)
        if not has_event_title and not has_handling_cue:
            continue
        if not any(word in haystack for word in ("报名", "人数", "名额", "过多", "太多", "截止", "调剂", "分组")):
            continue
        key = (
            str(source.get("board") or ""),
            str(source.get("tid") or source.get("title") or ""),
            str(source.get("floor") or ""),
        )
        if key in existing_keys:
            continue
        exact_score = 0.72 + keyword_boost(query, row)
        if all(term in title for term in event_terms):
            exact_score += 0.14
        if any(word in title for word in ("报名", "人数", "名额", "名单")):
            exact_score += 0.12
        if has_handling_cue:
            exact_score += 0.08
        exact.append(
            {
                "id": row.get("id") or idx,
                "score": 0.0,
                "rerank_score": exact_score,
                "text": text,
                "source": source,
            }
        )
        existing_keys.add(key)
        if len(exact) >= max(top_k * 2, 10):
            break

    if not exact:
        return ranked
    combined = exact + ranked
    combined.sort(key=lambda item: item["rerank_score"], reverse=True)
    return combined


def info() -> dict[str, Any]:
    meta_count = 0
    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as f:
            meta_count = sum(1 for line in f if line.strip())
    index_size = INDEX_PATH.stat().st_size if INDEX_PATH.exists() else 0
    return {
        "root": str(ROOT),
        "index": str(INDEX_PATH),
        "meta": str(META_PATH),
        "model": MODEL_NAME,
        "meta_count": meta_count,
        "index_bytes": index_size,
        "ready": INDEX_PATH.exists() and META_PATH.exists() and meta_count > 0,
    }


def fix_url(source: dict[str, Any]) -> str:
    """Generate a correct, clickable URL using p=1.
    The stored URL uses (floor-1)//12+1 for page number, which is unreliable
    because floor numbers in the metadata are inflated and don't match
    the actual pagination of each thread. Always link to p=1 instead."""
    bid = source.get("bid", 1)
    tid = source.get("tid", 0)
    base = f"https://www.chexie.net/bbs/content/index.php?bid={bid}&tid={tid}&p=1"
    floor = source.get("floor")
    if floor:
        return f"{base}#第{floor}楼"
    return base


def cite(source: dict[str, Any]) -> str:
    label = f"｜{source.get('source_label')}" if source.get("source_label") else ""
    url = fix_url(source)
    return (
        f"【{source.get('board', '未知版面')}】《{source.get('title', '未知标题')}》"
        f"第{source.get('floor', '?')}楼{label}\n"
        f"  链接: {url}"
    )


def print_human(payload: dict[str, Any]) -> None:
    print(f"查询：{payload['question']}")
    print(f"索引：{payload['count']} 条 evidence｜模型：{payload['model']}\n")
    for i, item in enumerate(payload["results"], start=1):
        source = item["source"]
        print(
            f"━━━ 结果 {i}/{len(payload['results'])} "
            f"score={item['score']:.4f} rerank={item['rerank_score']:.4f} ━━━"
        )
        print(cite(source))
        print(f"作者: {source.get('author')} | 时间: {source.get('time')}")
        print(item["text"][:1200].strip())
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Peking University Cycling Association KB.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--pipeline", "-p", action="store_true",
                        help="Use enhanced pipeline (deep search + content filter)")
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Generate structured topic analysis (powers --pipeline)")
    parser.add_argument("--bm25", action="store_true",
                        help="Enable BM25 keyword search (requires ~15s init)")
    parser.add_argument("--fts5", action="store_true",
                        help="Use FTS5 SQLite LIKE search (author/keyword/time)")
    parser.add_argument("--author", type=str, help="Author filter for --fts5")
    parser.add_argument("--time-start", type=str, help="Time start for --fts5 (YYYY-MM-DD)")
    parser.add_argument("--time-end", type=str, help="Time end for --fts5 (YYYY-MM-DD)")
    parser.add_argument("--board", type=str, help="Board filter for --fts5")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Auto-detect analyze for question-type queries
    qtype = detect_query_type(args.query or "")

    if args.fts5:
        keywords = [args.query] if args.query else None
        results = fts5_search(
            keywords=keywords, author=args.author,
            time_start=args.time_start, time_end=args.time_end,
            board=args.board, limit=args.top_k or 30,
        )
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for i, r in enumerate(results, 1):
                src = r['source']
                print(f'[{i}] {src["board"]} | {src["title"][:40]} | {src["author"]} | 第{src["floor"]}楼 | {src["post_time"]}')
                print(f'    {r["text"][:300]}')
                print()
        return

    if args.pipeline or args.analyze or qtype in ("fuzzy", "mixed"):
        from pipeline import search_pipeline
        if not args.query:
            parser.error("query required")
        use_analyze = args.analyze or (args.query and any(
            kw in args.query for kw in ("分析", "总结", "怎么回事", "怎么看", "讨论", "争议")))
        result = search_pipeline(args.query, top_k=args.top_k or 20,
                                  use_bm25=args.bm25, verbose=args.verbose,
                                  analyze=use_analyze)
        print(result["output"])
        print(f"\n  [{result['time']}s]")
        return

    if args.info:
        print(json.dumps(info(), ensure_ascii=False, indent=2))
        return
    if not args.query:
        parser.error("query required unless --info")

    payload = search(args.query, args.top_k, args.candidate_k)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human(payload)


if __name__ == "__main__":
    main()
