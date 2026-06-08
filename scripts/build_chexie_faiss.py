#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BOARD_NAMES = {1: "车协工作区", 2: "行者足音", 3: "车友宝典", 4: "纯净水", 7: "一技之长"}
RAW_DIRS = {
    1: "/home/ubuntu/workspace/chexie_data_bid1/threads.json",
    2: "/home/ubuntu/workspace/chexie_data_bid2/threads.json",
    3: "/home/ubuntu/workspace/chexie_data_bid3/threads.json",
    4: "/home/ubuntu/workspace/chexie_data_bid4/threads.json",
    7: "/home/ubuntu/workspace/chexie_data_bid7/threads.json",
}
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BACKEND = "torch"
MODEL_DIR = "/opt/wiki/models"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\ufffd", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def floor_int(value) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 1


def source_label(title: str, text: str) -> str:
    haystack = f"{title}\n{text[:800]}"
    patterns = [
        r"第[一二三四五六七八九十百零〇0-9]+次[^，。\n]{0,12}执委会",
        r"[0-9]{4}[^，。\n]{0,12}执委会",
        r"(春季|秋季|暑期|寒假)[^，。\n]{0,8}执委会",
        r"理事会[^，。\n]{0,24}(通知|决议|任命|提醒)",
        r"财务(制度|总结|公开|报销)",
    ]
    for pattern in patterns:
        match = re.search(pattern, haystack)
        if match:
            return match.group(0)
    return ""


def cues(title: str, text: str) -> str:
    haystack = f"{title}\n{text}"
    mapping = {
        "proposal": ["建议", "提议", "提案", "方案", "是否", "拟"],
        "practice": ["流程", "规定", "安排", "执行", "报名", "分组", "制度", "规则"],
        "benefit": ["优点", "好处", "有利于", "便于", "提高", "避免", "保证"],
        "risk": ["问题", "风险", "缺点", "不足", "争议", "反对", "事故", "取消"],
        "outcome": ["结果", "总结", "复盘", "最终", "实际", "完成", "公示"],
    }
    return ",".join(k for k, words in mapping.items() if any(w in haystack for w in words))


def split_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Semantic-aware chunking: split at sentence boundaries (。？！.?!).

    - Text shorter than max_chars is returned as a single chunk (no splitting).
    - Longer text is split at sentence-ending punctuation, greedily packing
      sentences into chunks up to max_chars.
    - Each new chunk starts with the last `overlap` characters of the previous
      chunk to preserve context continuity.
    - If a single sentence exceeds max_chars, it falls back to character-level
      splitting for that sentence only.
    """
    if len(text) <= max_chars:
        return [text]

    # Split text into sentences at Chinese/English sentence-ending punctuation.
    # Keep the delimiter attached to the preceding sentence.
    sentences = re.split(r'(?<=[。？！.?!])\s*', text)
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        # Fallback: no sentence boundaries found, use fixed-length slicing
        chunks = []
        start = 0
        step = max(1, max_chars - overlap)
        while start < len(text):
            chunk = text[start : start + max_chars].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        # Handle a single sentence that is itself longer than max_chars
        if len(sentence) > max_chars:
            # Flush current chunk first
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Sub-split the long sentence at character boundaries
            start = 0
            step = max(1, max_chars - overlap)
            while start < len(sentence):
                part = sentence[start : start + max_chars].strip()
                if part:
                    chunks.append(part)
                start += step
            # Carry overlap from the last sub-chunk into next iteration
            if chunks and overlap > 0:
                last = chunks[-1]
                current_chunk = last[-overlap:] if len(last) > overlap else last
            continue

        # Check if adding this sentence would exceed max_chars
        if current_chunk and len(current_chunk) + len(sentence) > max_chars:
            chunks.append(current_chunk.strip())
            # Start next chunk with overlap from the tail of current_chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def iter_rows(min_chars: int, max_chars: int, overlap: int):
    for bid, raw_path in RAW_DIRS.items():
        board = BOARD_NAMES[bid]
        with open(raw_path, "r", encoding="utf-8") as f:
            threads = json.load(f)
        for thread in threads:
            tid = int(thread.get("tid") or 0)
            title = str(thread.get("title") or "").strip()
            posts = thread.get("posts") or []
            for post_index, post in enumerate(posts, start=1):
                text = normalize_text(post.get("content") or "")
                if len(text) < min_chars:
                    continue
                floor = floor_int(post.get("floor"))
                author = post.get("author") or ""
                time = post.get("time") or ""
                # Use p=1 always — floor-based page calculation is unreliable
                # because floor numbers don't match actual thread pagination.
                label = source_label(title, text)
                prefix = f"【{board}】《{title}》第{floor}楼｜{author}｜{time}"
                url = f"https://www.chexie.net/bbs/content/index.php?bid={bid}&tid={tid}&p=1"
                parts = split_text(text, max_chars, overlap)
                for chunk_index, part in enumerate(parts, start=1):
                    doc = f"{prefix}\n链接: {url}\n\n{part}"
                    yield {
                        "text": doc,
                        "source": {
                            "bid": bid,
                            "board": board,
                            "tid": tid,
                            "title": title,
                            "floor": floor,
                            "post_index": post_index,
                            "chunk_index": chunk_index,
                            "chunk_count": len(parts),
                            "author": author,
                            "time": time,
                            "url": url,
                            "source_label": label,
                            "cues": cues(title, text),
                        },
                    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/opt/chexie-knowledge")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "faiss_meta.jsonl"
    index_path = out_dir / "faiss_index.bin"

    rows = list(iter_rows(args.min_chars, args.max_chars, args.overlap))
    print(f"rows={len(rows)}")
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_DIR, backend=BACKEND)
    dim = model.get_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    with meta_path.open("w", encoding="utf-8") as meta_file:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = [row["text"] for row in batch]
            vecs = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False)
            vecs = np.asarray(vecs, dtype="float32")
            index.add(vecs)
            for row in batch:
                meta_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if start % (args.batch_size * 20) == 0:
                print(f"embedded={start + len(batch)}/{len(rows)}")

    faiss.write_index(index, str(index_path))
    info = {
        "model": MODEL_NAME,
        "count": len(rows),
        "dim": dim,
        "index": str(index_path),
        "meta": str(meta_path),
    }
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    # === Build title index ===
    print("\n--- Building title index ---")
    title_meta: list[dict] = []
    title_texts: list[str] = []
    seen_titles: set[int] = set()
    for bid, raw_path in RAW_DIRS.items():
        board = BOARD_NAMES[bid]
        with open(raw_path, "r", encoding="utf-8") as f:
            threads = json.load(f)
        for thread in threads:
            tid = int(thread.get("tid") or 0)
            if tid in seen_titles:
                continue
            seen_titles.add(tid)
            title = str(thread.get("title") or "").strip()
            if not title:
                continue
            title_texts.append(f"【{board}】{title}")
            title_meta.append({
                "tid": tid,
                "bid": bid,
                "board": board,
                "title": title,
            })

    title_index = faiss.IndexFlatIP(dim)
    for start in range(0, len(title_texts), args.batch_size):
        batch_texts = title_texts[start:start + args.batch_size]
        vecs = model.encode(batch_texts, batch_size=args.batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        title_index.add(vecs)

    title_idx_path = out_dir / "title_index.bin"
    title_meta_path = out_dir / "title_meta.jsonl"
    faiss.write_index(title_index, str(title_idx_path))
    with title_meta_path.open("w", encoding="utf-8") as f:
        for item in title_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"title_index: {title_index.ntotal} vectors -> {title_idx_path}")
    print(f"title_meta: {len(title_meta)} entries -> {title_meta_path}")


if __name__ == "__main__":
    main()
