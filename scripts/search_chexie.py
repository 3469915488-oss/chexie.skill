#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys

REMOTE = "ubuntu@49.232.25.231"
REMOTE_SCRIPT = "/opt/chexie-knowledge/search_chexie.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the remote Chexie knowledge base.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    remote_args = [REMOTE_SCRIPT]
    if args.info:
        remote_args.append("--info")
    else:
        if not args.query:
            parser.error("query required unless --info")
        remote_args.extend([args.query, "--top-k", str(args.top_k)])
        if args.candidate_k is not None:
            remote_args.extend(["--candidate-k", str(args.candidate_k)])
        if args.json:
            remote_args.append("--json")

    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        REMOTE,
        *remote_args,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
