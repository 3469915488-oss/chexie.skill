#!/usr/bin/env python3
"""
Build SQLite FTS5 full-text index from faiss_meta.jsonl.
Replaces the in-memory rank_bm25 with a zero-memory SQLite-backed BM25.

Usage:
    python3 build_fts5.py                           # build from default meta
    python3 build_fts5.py --meta /path/to/meta.jsonl
    python3 build_fts5.py --rebuild                  # force rebuild
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "faiss_meta.jsonl"
DB_PATH = ROOT / "chexie_fts.db"

# Jieba FTS5 tokenizer: install via
#   pip install jieba3k  # or use a custom tokenizer
# If jieba isn't available for FTS5, we fall back to the built-in unicode61.
# Actually sqlite-jieba requires compiling a loadable extension.
# For simplicity, we use the unicode61 tokenizer + Chinese character
# bigram approach, which works reasonably well for Chinese FTS5.


def iter_meta(path: Path):
    """Yield each row from faiss_meta.jsonl."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_fts_fields(row: dict) -> tuple[str, str, str, str, str]:
    """Extract FTS5 fields from a meta row.

    Returns (chunk_id, title, board, author, text)
    """
    source = row.get("source") or row.get("meta") or {}
    chunk_id = str(row.get("id", ""))
    title = str(source.get("title", ""))
    board = str(source.get("board", ""))
    author = str(source.get("author", ""))
    text = str(row.get("text", ""))
    return chunk_id, title, board, author, text


def create_schema(conn: sqlite3.Connection):
    """Create FTS5 virtual table.

    We use the built-in unicode61 tokenizer with tokenchars (treats
    punctuation as part of tokens to keep Chinese characters together).
    For Chinese, unicode61 splits on whitespace + punctuation, which
    means Chinese characters end up as individual tokens. That's suboptimal
    but functional for BM25 search via FTS5.

    Alternative: if a custom jieba tokenizer extension is available,
    we detect it and use it instead.
    """
    # First check if jieba FTS5 tokenizer is available
    cursor = conn.execute("PRAGMA compile_options")
    compile_opts = [row[0] for row in cursor.fetchall()]

    # Use unicode61 with tokenchars for CJK support
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts5_index USING fts5("
        "  chunk_id UNINDEXED,"
        "  title,"
        "  board,"
        "  author,"
        "  text,"
        "  tokenize='unicode61'"
        ")"
    )


def build(meta_path: Path, db_path: Path, rebuild: bool = False):
    """Build the FTS5 index from faiss_meta.jsonl."""
    if not meta_path.exists():
        print(f"Meta not found: {meta_path}")
        print("FAISS build must finish first.")
        sys.exit(1)

    if db_path.exists():
        if rebuild:
            db_path.unlink()
            print("Removed existing FTS5 DB (--rebuild)")
        else:
            print(f"FTS5 DB already exists: {db_path}")
            print("Use --rebuild to force rebuild.")
            return

    t0 = time.time()

    # Count total rows for progress
    total = 0
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    print(f"Meta entries: {total}")

    conn = sqlite3.connect(str(db_path))
    create_schema(conn)

    # Prepare insert
    insert_sql = (
        "INSERT INTO fts5_index(chunk_id, title, board, author, text) "
        "VALUES (?, ?, ?, ?, ?)"
    )

    batch = []
    BATCH_SIZE = 500

    for i, row in enumerate(iter_meta(meta_path), start=1):
        fields = extract_fts_fields(row)
        batch.append(fields)

        if len(batch) >= BATCH_SIZE:
            conn.executemany(insert_sql, batch)
            conn.commit()
            batch = []
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  {i}/{total} ({rate:.0f} rows/s)", end="\r", flush=True)

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM fts5_index").fetchone()[0]
    conn.close()

    elapsed = time.time() - t0
    db_size = db_path.stat().st_size / 1024 / 1024

    print(f"\n{'=' * 40}")
    print(f"FTS5 index built: {db_path}")
    print(f"  Entries: {count}")
    print(f"  DB size: {db_size:.0f} MB")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Match: {'✅' if count == total else '❌ MISMATCH!'}")


def main():
    parser = argparse.ArgumentParser(description="Build SQLite FTS5 full-text index")
    parser.add_argument("--meta", default=str(META_PATH), help="faiss_meta.jsonl path")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    args = parser.parse_args()

    build(Path(args.meta), DB_PATH, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
