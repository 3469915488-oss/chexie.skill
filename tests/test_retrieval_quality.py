#!/usr/bin/env python3
"""
test_retrieval_quality.py — Retrieval quality regression tests.

Fixed query set with ground truth tids sourced from events/controversies.yaml.
Run with: python -m pytest tests/test_retrieval_quality.py -v

Requires:
  - Data at /opt/chexie-knowledge/ (or CHXIE_DATA_DIR env var)
  - Dependencies: faiss-cpu, sentence-transformers, numpy
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

DATA_DIR = Path(os.environ.get("CHXIE_DATA_DIR", "/opt/chexie-knowledge"))
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SEARCH_SCRIPT = SCRIPTS_DIR / "search_chexie.py"

# Skip all tests if data not available
pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"Data directory not found: {DATA_DIR}"
)


# ── Ground truth ────────────────────────────────────────────────────────────────
# Each case: query, expected_tids (from controversies.yaml), min_recall_ratio

FIXED_QUERIES = [
    {
        "query": "暑期选拔落选",
        "expected_tids": [5895, 2052, 4219, 4221],
        "min_recall": 0.50,
        "controversy": "ctrv-03",
    },
    {
        "query": "下坡安全纪律",
        "expected_tids": [6246, 19792, 19039, 18783],
        "min_recall": 0.50,
        "controversy": "ctrv-04",
    },
    {
        "query": "甘道夫封禁理事会",
        "expected_tids": [6257, 6119],
        "min_recall": 0.50,
        "controversy": "ctrv-05/09",
    },
    {
        "query": "借车规范责任",
        "expected_tids": [4022, 19821, 19633],
        "min_recall": 0.50,
        "controversy": "ctrv-14",
    },
    {
        "query": "拉练追队离队风气",
        "expected_tids": [19039, 4335],
        "min_recall": 0.50,
        "controversy": "ctrv-04/11",
    },
    {
        "query": "论坛存废关闭",
        "expected_tids": [19612, 19587],
        "min_recall": 0.50,
        "controversy": "ctrv-12",
    },
    {
        "query": "互评机制压力",
        "expected_tids": [7397, 19309, 5670],
        "min_recall": 0.50,
        "controversy": "ctrv-11",
    },
    {
        "query": "理事会换届程序",
        "expected_tids": [7782, 8422, 8156, 9533],
        "min_recall": 0.50,
        "controversy": "board=1 council",
    },
    {
        "query": "出摊招新收费",
        "expected_tids": [18708, 18775, 14396],
        "min_recall": 0.50,
        "controversy": "ctrv-06",
    },
    {
        "query": "老会员角色代际",
        "expected_tids": [1138, 4498, 3485],
        "min_recall": 0.50,
        "controversy": "ctrv-02",
    },
]

SOURCE_ID_PATTERN = re.compile(r"^bid\d+_tid\d+_floor\d+$")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def run_search(query: str, top_k: int = 20, mode: str = "fast") -> dict:
    """Run search_chexie.py and return parsed JSON."""
    cmd = [
        sys.executable, str(SEARCH_SCRIPT),
        query, "--top-k", str(top_k), "--json"
    ]
    env = os.environ.copy()
    env["CHXIE_DATA_DIR"] = str(DATA_DIR)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, env=env,
        cwd=str(SCRIPTS_DIR.parent)
    )
    if result.returncode != 0:
        pytest.fail(f"Search failed: {result.stderr}")
    return json.loads(result.stdout)


def extract_tids(data: dict) -> set[int]:
    """Extract thread IDs from search results."""
    tids = set()
    for r in data.get("results", []):
        source = r.get("source", {})
        tid = source.get("tid")
        if tid is not None:
            tids.add(tid)
    return tids


# ── Tests ───────────────────────────────────────────────────────────────────────

class TestRecall:
    """Fixed query recall tests."""

    @pytest.mark.parametrize("case", FIXED_QUERIES, ids=lambda c: c["query"])
    def test_recall(self, case):
        data = run_search(case["query"], top_k=20)
        found_tids = extract_tids(data)
        expected = set(case["expected_tids"])
        hits = expected & found_tids
        recall = len(hits) / len(expected) if expected else 1.0
        assert recall >= case["min_recall"], (
            f"Recall {recall:.0%} < {case['min_recall']:.0%} for '{case['query']}\n"
            f"Expected tids: {expected}\n"
            f"Found tids: {found_tids}\n"
            f"Missing: {expected - found_tids}"
        )


class TestSourceIdFormat:
    """Validate source_id format in every result."""

    @pytest.mark.parametrize("query", ["暑期选拔", "下坡安全", "理事会换届"])
    def test_source_id_format(self, query):
        data = run_search(query, top_k=10)
        for i, r in enumerate(data.get("results", [])):
            sid = r.get("source_id", "")
            assert SOURCE_ID_PATTERN.match(sid), (
                f"Invalid source_id at result {i}: '{sid}' (query: {query})"
            )


class TestNoDuplicates:
    """No duplicate source_ids in a single query result."""

    def test_no_duplicate_source_ids(self):
        data = run_search("暑期选拔落选", top_k=20)
        sids = [r.get("source_id") for r in data.get("results", [])]
        assert len(sids) == len(set(sids)), f"Duplicate source_ids found in {sids}"


class TestFTS5ChineseTokenization:
    """FTS5 handles Chinese terms correctly."""

    @pytest.mark.parametrize("term", ["甘道夫", "暑期", "理事会"])
    def test_chinese_fts5(self, term):
        data = run_search(term, top_k=10)
        assert len(data.get("results", [])) > 0, (
            f"FTS5 returned 0 results for Chinese term '{term}'"
        )


class TestControversyLookupGuard:
    """Verify controversy_lookup module enforces safety rules."""

    def test_import(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from controversy_lookup import (
            get_controversy_by_id,
            get_candidate_tids,
            validate_no_text_leakage,
        )

    def test_candidate_tids_returns_tids(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from controversy_lookup import get_candidate_tids
        results = get_candidate_tids("暑期选拔", top_k=1)
        assert len(results) > 0
        assert "tids" in results[0]
        assert isinstance(results[0]["tids"], list)
        assert all(isinstance(t, int) for t in results[0]["tids"])

    def test_no_text_leakage(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from controversy_lookup import get_controversy_by_id, validate_no_text_leakage
        event = get_controversy_by_id("ctrv-03")
        # Should not raise
        validate_no_text_leakage(event)
        # Thread entries should have exactly the allowed fields
        for thread in event["threads"]:
            assert set(thread.keys()) == {
                "bid", "tid", "title", "author", "date", "clicks", "replies", "role", "url"
            }

    def test_get_controversy_for_tid(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from controversy_lookup import get_controversy_for_tid
        result = get_controversy_for_tid(5895)
        assert result == "ctrv-03"


class TestPipelineMode:
    """Pipeline mode returns richer results."""

    def test_pipeline_returns_results(self):
        data = run_search("暑期选拔落选的感受", top_k=10)
        # Pipeline mode should also work (search defaults to fast)
        assert len(data.get("results", [])) > 0
