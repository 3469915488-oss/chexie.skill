#!/usr/bin/env python3
"""
Build BM25 index cache for chexie-knowledge.
Run once:  python3 build_bm25.py
Takes ~2.5 minutes, creates bm25_index.jsonl in the chexie-knowledge directory.
"""

import json
import re
import sys
import time
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "faiss_meta.jsonl"
OUT_PATH = ROOT / "bm25_index.jsonl"


def main():
    if OUT_PATH.exists():
        print(f"BM25 cache already exists: {OUT_PATH}")
        print(f"Delete it first if you want to rebuild.")
        return

    t0 = time.time()
    rows = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} entries. Tokenizing with jieba...")

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(rows):
            text = item.get("text", "")
            clean = re.sub(r'\s+', ' ', text)
            tokens = list(jieba.cut(clean))
            fout.write(json.dumps({
                "id": item.get("id", ""),
                "text": clean,
                "tokens": tokens,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 20000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(rows)} ({elapsed:.0f}s, ~{elapsed/(i+1)*len(rows):.0f}s total)")

    print(f"Done! {len(rows)} docs in {time.time()-t0:.0f}s")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
