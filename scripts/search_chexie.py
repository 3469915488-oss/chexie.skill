#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path("/opt/chexie-knowledge")
INDEX_PATH = ROOT / "faiss_index.bin"
META_PATH = ROOT / "faiss_meta.jsonl"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_PATH = Path(
    "/opt/wiki/models/models--BAAI--bge-small-zh-v1.5/"
    "snapshots/7999e1d3359715c523056ef9478215996d62a620"
)
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
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Auto-detect analyze for question-type queries
    qtype = detect_query_type(args.query or "")

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
