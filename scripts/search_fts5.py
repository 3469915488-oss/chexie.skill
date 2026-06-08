#!/usr/bin/env python3
"""
FTS5 search backend for chexie-knowledge.
Zero-memory BM250index backed by SQLite FTS5.

Usage (programmatic):
    from search_fts5 import fts5_search
    results = fts5_search("白河 扎胎", top_k=30)

CLI:
    python3 search_fts5.py "白河 扎胎"
    python3 search_fts5.py "押后 AND 扎胎" --top-k 20
"""

import json
import os
import re
import sqlite3
import yaml
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "chexie_fts.db"
JARGON_PATH = ROOT / "jargon.yaml"

# ── Jieba integration (optional, for FTS5 tokenization) ──
# If jieba_tokenizer extension is available, query building is simpler.
# Otherwise we rely on character-based matching via unicode61 tokenchars.

# ── Black话 loading ──
_JARGONS: dict[str, dict[str, Any]] | None = None


def load_jargons() -> dict[str, dict[str, Any]]:
    global _JARGONS
    if _JARGONS is not None:
        return _JARGONS
    if JARGON_PATH.exists():
        with JARGON_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            _JARGONS = data.get("jargons", {})
    else:
        _JARGONS = {}
    return _JARGONS


# ── Query expansion ──

def expand_query(query: str) -> str:
    """将黑话替换为搜索扩展词。
    例: "押后出了什么问题" → "(押后 OR 修车 OR 补胎) AND 问题"
    """
    jargons = load_jargons()
    expanded = query
    for jargon, config in jargons.items():
        if jargon in expanded:
            expand_terms = config.get("search_expand", [jargon])
            # Build OR group. Keep original term first.
            or_group = " OR ".join(f'"{t}"' for t in expand_terms)
            # Only replace the first occurrence to avoid over-matching
            expanded = expanded.replace(jargon, f"({or_group})", 1)
    return expanded


def build_fts5_query(query: str) -> str:
    """将用户查询转换为 FTS5 query 语法。

    处理：
    1. 空格转换为 AND（FTS5 默认是 AND）
    2. 如果用户已使用 AND/OR/NOT，保留
    3. 中文短语加引号
    """
    # If the query already has boolean operators, leave as-is
    if re.search(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE):
        return query

    # Otherwise, treat space-separated terms as AND
    terms = query.strip().split()
    if len(terms) <= 1:
        return query

    # Wrap each term in double quotes (better for Chinese)
    quoted = [f'"{t}"' if not t.startswith('"') else t for t in terms]
    return " AND ".join(quoted)


# ── Stop words ──

STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "什么", "怎么", "为什么", "因为", "所以",
    "但是", "如果", "虽然", "而且", "然后", "所以", "可以", "可能",
    "这个", "那个", "这些", "那些", "已经", "还是", "或者", "因为",
    "不过", "只是", "就是", "不是", "没有", "还是", "但是",
    "吗", "吧", "呢", "啊", "哦", "嗯", "呀", "么",
}


def filter_stop_words(query: str) -> str:
    """Remove stop words from query (but keep quoted phrases intact)."""
    terms = query.split()
    filtered = [t for t in terms if t.lower() not in STOP_WORDS and len(t) > 1]
    return " ".join(filtered) if filtered else query


# ── FTS5 search ──

# Connection cache
_conn: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection | None:
    global _conn
    if _conn is not None:
        return _conn
    if not DB_PATH.exists():
        return None
    _conn = sqlite3.connect(str(DB_PATH))
    return _conn


def fts5_search(query: str, top_k: int = 30) -> list[dict[str, Any]]:
    """Search the FTS5 full-text index.

    Args:
        query: user query (plain text, with or without jargon)
        top_k: max results

    Returns:
        List of dicts with keys: chunk_id, score, text, title, board, author, url
    """
    conn = get_connection()
    if conn is None:
        # No FTS5 index yet, return empty
        return []

    # Step 1: clean query
    clean_q = filter_stop_words(query.strip())

    # Step 2: expand jargon
    expanded_q = expand_query(clean_q)

    # Step 3: build FTS5 query
    fts5_q = build_fts5_query(expanded_q)

    if not fts5_q:
        return []

    # Step 4: execute FTS5 search
    try:
        cursor = conn.execute(
            "SELECT chunk_id, title, board, author, text, rank "
            "FROM fts5_index WHERE text MATCH ? "
            "ORDER BY rank LIMIT ?",
            (fts5_q, top_k),
        )
    except sqlite3.OperationalError as e:
        # FTS5 might fail on certain query syntax, fall back to LIKE
        print(f"[fts5] MATCH failed: {e}", file=__import__('sys').stderr)
        print(f"[fts5] Query: {fts5_q}", file=__import__('sys').stderr)
        return _like_fallback(clean_q, top_k)

    results = []
    for row in cursor.fetchall():
        chunk_id, title, board, author, text, rank = row
        # rank -1 * bm25_score, so negate for intuitive ordering
        score = -float(rank) if rank < 0 else 0.0
        results.append({
            "chunk_id": chunk_id,
            "score": round(score, 4),
            "text": text,
            "title": title,
            "board": board,
            "author": author,
        })

    return results


def _like_fallback(query: str, top_k: int) -> list[dict[str, Any]]:
    """Fallback: use LIKE when FTS5 MATCH fails (less accurate but works)."""
    conn = get_connection()
    if conn is None:
        return []

    terms = [t.strip('"') for t in query.split() if len(t.strip('"')) > 1]
    if not terms:
        return []

    # Build dynamic LIKE conditions
    conditions = " AND ".join(f"text LIKE '%{t}%'" for t in terms)
    sql = f"SELECT chunk_id, title, board, author, text FROM fts5_index WHERE {conditions} LIMIT ?"

    try:
        cursor = conn.execute(sql, (top_k,))
    except sqlite3.OperationalError:
        return []

    results = []
    for row in cursor.fetchall():
        chunk_id, title, board, author, text = row
        results.append({
            "chunk_id": chunk_id,
            "score": 0.5,  # flat score for LIKE results
            "text": text,
            "title": title,
            "board": board,
            "author": author,
        })
    return results


# ── CLI ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTS5 search for chexie-knowledge")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = fts5_search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"FTS5 search: {args.query}")
        print(f"Results: {len(results)}\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. [{r['board']}] {r['title']} (score={r['score']:.4f})")
            print(f"   {r['author']}")
            print(f"   {r['text'][:200]}...")
            print()
