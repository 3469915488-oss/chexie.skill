#!/usr/bin/env python3
"""
controversy_lookup.py — Safe wrapper for controversy event index.

Controversy index is a SECONDARY source. It provides candidate tids for further
retrieval. Never use controversy summaries as facts — always retrieve original
post chunks via search_chexie.py.

This module enforces the rule: controversy event index can ONLY return candidate
thread IDs, never full text content.

Usage (CLI):
    python scripts/controversy_lookup.py "暑期选拔"
    python scripts/controversy_lookup.py "创会精神" --top-k 5

Usage (Python):
    from controversy_lookup import get_controversy_by_id, get_candidate_tids
    event = get_controversy_by_id("ctrv-01")
    candidates = get_candidate_tids("暑期选拔", top_k=3)
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
CONTROVERSIES_PATH = ROOT / "events" / "controversies.yaml"

# Fields allowed in thread entries — NO text/body/content fields
ALLOWED_THREAD_FIELDS = frozenset({
    "bid", "tid", "title", "author", "date", "clicks", "replies", "role", "url"
})

# ── Data loading ───────────────────────────────────────────────────────────────

_CONTROVERSIES: list[dict] | None = None


def _load_controversies() -> list[dict]:
    """Load and cache controversies from YAML."""
    global _CONTROVERSIES
    if _CONTROVERSIES is None:
        if not CONTROVERSIES_PATH.exists():
            raise FileNotFoundError(
                f"Controversies file not found: {CONTROVERSIES_PATH}"
            )
        with open(CONTROVERSIES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _CONTROVERSIES = data.get("controversies", [])
    return _CONTROVERSIES


def _sanitize_thread(thread: dict) -> dict:
    """
    Strip any fields not in ALLOWED_THREAD_FIELDS.
    Ensures no post text/body/summary leaks into thread entries.
    """
    return {k: v for k, v in thread.items() if k in ALLOWED_THREAD_FIELDS}


def _sanitize_threads(threads: list[dict]) -> list[dict]:
    """Sanitize a list of thread dicts."""
    return [_sanitize_thread(t) for t in threads]


# ── Public API ─────────────────────────────────────────────────────────────────


def get_controversy_by_id(ctrv_id: str) -> dict:
    """
    Get a controversy event by its ID (e.g. 'ctrv-01').

    Returns event metadata (name, period, summary, tags, viewpoints, outcomes)
    and a list of candidate thread entries. Thread entries contain ONLY:
    bid, tid, title, author, date, clicks, replies, role, url.

    Does NOT include any post text/body/content.

    Raises:
        KeyError: If the controversy ID is not found.
    """
    controversies = _load_controversies()
    for ctrv in controversies:
        if ctrv.get("id") == ctrv_id:
            result = {
                "id": ctrv.get("id"),
                "name": ctrv.get("name"),
                "period": ctrv.get("period"),
                "intensity": ctrv.get("intensity"),
                "tags": ctrv.get("tags", []),
                "summary": ctrv.get("summary", ""),
                "threads": _sanitize_threads(ctrv.get("threads", [])),
                "viewpoints": ctrv.get("viewpoints", []),
                "outcomes": ctrv.get("outcomes", []),
                "related_controversies": ctrv.get("related_controversies", []),
            }
            validate_no_text_leakage(result)
            return result

    raise KeyError(f"Controversy not found: {ctrv_id}")


def get_candidate_tids(query: str, top_k: int = 3) -> list[dict]:
    """
    Fuzzy match query against controversy names, summaries, and tags.
    Return candidate tids from the top matching events.

    Args:
        query: Search string (e.g. '暑期选拔', '创会精神').
        top_k: Number of top matching events to return tids from.

    Returns:
        List of dicts, each with:
            - ctrv_id: controversy ID
            - ctrv_name: controversy name
            - score: match score (0.0–1.0)
            - tids: list of candidate thread IDs (ints)
            - threads: sanitized thread entries (metadata only)
    """
    controversies = _load_controversies()
    query_lower = query.lower()

    scored = []
    for ctrv in controversies:
        # Build a match string from name + summary + tags
        name = (ctrv.get("name") or "").lower()
        summary = (ctrv.get("summary") or "").lower()
        tags = " ".join(ctrv.get("tags", [])).lower()
        match_text = f"{name} {summary} {tags}"

        # SequenceMatcher fuzzy ratio
        score = SequenceMatcher(None, query_lower, match_text).ratio()

        # Also check substring match (boost)
        if query_lower in match_text:
            score = max(score, 0.85)

        # Check individual tag exact match
        for tag in ctrv.get("tags", []):
            if query_lower == tag.lower():
                score = max(score, 0.95)
            elif query_lower in tag.lower() or tag.lower() in query_lower:
                score = max(score, 0.80)

        scored.append((score, ctrv))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, ctrv in scored[:top_k]:
        threads = _sanitize_threads(ctrv.get("threads", []))
        tids = [t["tid"] for t in threads if "tid" in t]
        results.append({
            "ctrv_id": ctrv.get("id"),
            "ctrv_name": ctrv.get("name"),
            "score": round(score, 4),
            "tids": tids,
            "threads": threads,
        })

    # Validate all results
    for r in results:
        validate_no_text_leakage(r)

    return results


def get_controversy_for_tid(tid: int) -> str | None:
    """
    Given a thread ID (tid), return which controversy it belongs to (if any).

    Args:
        tid: Thread ID (integer).

    Returns:
        The controversy ID (e.g. 'ctrv-01') if found, else None.
    """
    controversies = _load_controversies()
    for ctrv in controversies:
        for thread in ctrv.get("threads", []):
            if thread.get("tid") == tid:
                return ctrv.get("id")
    return None


# ── Guard function ─────────────────────────────────────────────────────────────


def validate_no_text_leakage(result: dict) -> bool:
    """
    Assert no field in the result dict contains text longer than 200 chars.
    This catches accidental inclusion of post bodies or long-form content.

    The summary field is exempted (it's event-level metadata, not post content).
    Thread-level fields like 'title' are naturally short and should never exceed
    200 chars.

    Raises:
        ValueError: If any non-exempt field exceeds 200 chars.

    Returns:
        True if validation passes.
    """
    EXEMPT_KEYS = {"summary", "stance", "outcomes"}
    MAX_LENGTH = 200

    def _check(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _check(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _check(item, f"{path}[{i}]")
        elif isinstance(obj, str):
            # Skip exempt keys (by field name, not path)
            field_name = path.rsplit(".", 1)[-1].rsplit("[", 1)[0]
            if field_name in EXEMPT_KEYS:
                return
            if len(obj) > MAX_LENGTH:
                raise ValueError(
                    f"TEXT LEAKAGE DETECTED at '{path}': "
                    f"string length {len(obj)} exceeds {MAX_LENGTH} chars. "
                    f"Preview: {obj[:80]}..."
                )

    _check(result)
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controversy index lookup — returns candidate tids only (no post text)."
    )
    parser.add_argument("query", help="Search query (e.g. '暑期选拔', '创会精神')")
    parser.add_argument("--top-k", type=int, default=3, help="Top K matching events (default: 3)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = get_candidate_tids(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if not results:
            print(f"No controversies matched: '{args.query}'")
            sys.exit(0)

        print(f"Query: '{args.query}' — {len(results)} matching event(s)\n")
        print("=" * 70)
        for r in results:
            print(f"\n[{r['ctrv_id']}] {r['ctrv_name']}  (score: {r['score']})")
            print("-" * 50)
            print(f"  Candidate tids: {r['tids']}")
            for t in r["threads"]:
                print(f"    - tid={t.get('tid')} | {t.get('title')}")
                print(f"      author={t.get('author')} date={t.get('date')} "
                      f"role={t.get('role')} clicks={t.get('clicks')}")
                print(f"      url={t.get('url')}")
        print("\n" + "=" * 70)
        print("⚠  Controversy index is SECONDARY. Use these tids with search_chexie.py")
        print("   to retrieve original post chunks. Never use summaries as facts.")


if __name__ == "__main__":
    main()
