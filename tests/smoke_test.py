#!/usr/bin/env python3
"""
chexie-knowledge smoke test
覆盖 5 个核心场景，确保 CLI 入口不退化。

用法:
    cd chexie-repo/
    CHEXIE_ROOT=/opt/chexie-knowledge python tests/smoke_test.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "search_chexie.py"
PASS = 0
FAIL = 0


def run(label: str, args: list[str], expect_json: bool = False,
        min_output_bytes: int = 50, timeout: int = 120) -> bool:
    global PASS, FAIL
    cmd = [sys.executable, str(SCRIPT)] + args
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"  FAIL  {label}  (timeout {timeout}s)")
        FAIL += 1
        return False

    elapsed = time.time() - t0
    ok = True

    # Check exit code
    if proc.returncode != 0:
        print(f"  FAIL  {label}  exit={proc.returncode}  ({elapsed:.1f}s)")
        if proc.stderr:
            print(f"        stderr: {proc.stderr[:300]}")
        ok = False

    # Check output not empty
    if len(proc.stdout) < min_output_bytes:
        print(f"  FAIL  {label}  output too short ({len(proc.stdout)} bytes)")
        ok = False

    # Check no traceback
    if "Traceback" in proc.stdout or "Traceback" in proc.stderr:
        print(f"  FAIL  {label}  traceback detected")
        if proc.stderr:
            print(f"        {proc.stderr[:500]}")
        ok = False

    # Optional JSON validation
    if expect_json and ok:
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(f"  FAIL  {label}  invalid JSON output")
            ok = False

    if ok:
        print(f"  PASS  {label}  ({elapsed:.1f}s, {len(proc.stdout)} bytes)")
        PASS += 1
    else:
        FAIL += 1
    return ok


def main():
    print(f"\n  chexie-knowledge smoke test")
    print(f"  script: {SCRIPT}")
    print(f"  python: {sys.executable}\n")

    # Test 1: --info (no heavy deps needed)
    run("--info 状态查询", ["--info"], expect_json=True)

    # Test 2: 快速搜索 (basic vector search)
    run("快速搜索 '春训时间'", ["春训时间", "--top-k", "3"])

    # Test 3: FTS5 搜索
    run("FTS5 搜索 '白河'", ["--fts5", "白河", "--top-k", "5"])

    # Test 4: Pipeline 深度复盘
    run("Pipeline 深度复盘",
        ["--pipeline", "25双日白河A组出现了什么问题", "--top-k", "10"],
        timeout=180)

    # Test 5: 快速搜索 JSON 输出
    run("JSON 输出", ["远征报名流程", "--top-k", "3", "--json"],
        expect_json=True)

    # Summary
    total = PASS + FAIL
    print(f"\n  {'='*40}")
    print(f"  {PASS}/{total} passed, {FAIL} failed\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
