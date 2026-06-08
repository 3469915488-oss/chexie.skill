#!/usr/bin/env python3
"""Build title index only (FAISS content index must exist).
Skips the full rebuild — reuses existing faiss_index.bin."""

import json, os, time, sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path("/opt/chexie-knowledge")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_DIR = "/opt/wiki/models"

BOARD_NAMES = {1: "车协工作区", 2: "行者足音", 3: "车友宝典", 4: "纯净水", 7: "一技之长"}
RAW_DIRS = {
    1: "/home/ubuntu/workspace/chexie_data_bid1/threads.json",
    2: "/home/ubuntu/workspace/chexie_data_bid2/threads.json",
    3: "/home/ubuntu/workspace/chexie_data_bid3/threads.json",
    4: "/home/ubuntu/workspace/chexie_data_bid4/threads.json",
    7: "/home/ubuntu/workspace/chexie_data_bid7/threads.json",
}

OUT_TITLE_INDEX = ROOT / "title_index.bin"
OUT_TITLE_META = ROOT / "title_meta.jsonl"
BATCH_SIZE = 128

t0 = time.time()

print("Loading model...")
model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_DIR, device="cpu")
dim = model.get_embedding_dimension()
print(f"Model loaded (dim={dim}) in {time.time()-t0:.0f}s")

# Collect unique titles
print("Collecting unique titles...")
title_texts: list[str] = []
title_meta: list[dict] = []
seen: set[int] = set()
total_threads = 0

for bid in sorted(RAW_DIRS.keys()):
    board = BOARD_NAMES[bid]
    path = RAW_DIRS[bid]
    with open(path, "r", encoding="utf-8") as f:
        threads = json.load(f)
    count = 0
    for thread in threads:
        tid = int(thread.get("tid") or 0)
        if tid in seen:
            continue
        seen.add(tid)
        title = str(thread.get("title") or "").strip()
        if not title:
            continue
        title_texts.append(f"【{board}】{title}")
        title_meta.append({"tid": tid, "bid": bid, "board": board, "title": title})
        count += 1
    total_threads += len(threads)
    print(f"  bid={bid} ({board}): {count} unique titles / {len(threads)} total threads")

print(f"Total: {len(title_texts)} unique titles from {total_threads} threads ({time.time()-t0:.0f}s)")

# Encode
print(f"Encoding {len(title_texts)} titles (batch_size={BATCH_SIZE})...")
title_index = faiss.IndexFlatIP(dim)
for start in range(0, len(title_texts), BATCH_SIZE):
    batch = title_texts[start:start + BATCH_SIZE]
    vecs = model.encode(batch, batch_size=BATCH_SIZE, normalize_embeddings=True,
                        show_progress_bar=False)
    vecs = np.asarray(vecs, dtype="float32")
    title_index.add(vecs)
    if (start // BATCH_SIZE) % 5 == 0:
        elapsed = time.time() - t0
        pct = (start + len(batch)) / len(title_texts) * 100
        print(f"  {start+len(batch)}/{len(title_texts)} ({pct:.0f}%) {elapsed:.0f}s", flush=True)

# Save
faiss.write_index(title_index, str(OUT_TITLE_INDEX))
with open(OUT_TITLE_META, "w", encoding="utf-8") as f:
    for item in title_meta:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

elapsed = time.time() - t0
print(f"\nDone! {elapsed:.0f}s")
print(f"title_index: {title_index.ntotal} vectors -> {OUT_TITLE_INDEX} ({OUT_TITLE_INDEX.stat().st_size/1024/1024:.0f}MB)")
print(f"title_meta:  {len(title_meta)} entries -> {OUT_TITLE_META}")
