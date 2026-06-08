#!/usr/bin/env python3
"""Extract entities (routes, problems, roles) from chexie BBS threads.json files.
Outputs entity → tid mapping for search-time boosting."""

import json, re, sys, os
from pathlib import Path
from collections import defaultdict

# Known routes from chexie.net common routes
ROUTES = [
    "十渡", "妙峰", "禅房", "白河", "九龙山", "白羊沟", "潭柘寺",
    "凤凰岭", "八达岭", "慕田峪", "黄花城", "大杨山", "石花洞",
    "十三陵", "东方红", "戒台寺", "卢沟桥", "香山", "八达岭",
    "四海", "永宁", "解字石", "慈悲峪", "高崖口", "霞云岭",
    "红井路", "六石路", "百花山", "东指壶", "镇罗营",
]

# Known problem types
PROBLEMS = [
    "扎胎", "爆胎", "漏气", "蛇咬",
    "摔车", "蹭摔",
    "中暑", "抽筋", "低血糖", "晒伤",
    "链条断", "掉链", "链条卡",
    "刹车失灵", "蹭碟", "蹭圈", "刹车啸叫",
    "变速不准", "跳链", "变速卡",
    "断辐条", "偏摆", "龙跳", "拿龙",
    "异响", "共振",
    "脚踏松动", "中轴异响", "碗组松动",
    "货架断裂", "驮包松动",
]

ROLES = ["押后", "队医", "队长", "领队", "前站", "押后负责", "队医负责"]

YEARS_PATTERN = re.compile(r'(?:19|20)\d{2}|[１２３４５６７８９０]{2,4}')
BID_PATTERN = re.compile(r'\d{2}(?:春|秋|暑|寒|冬)')

OUT_DIR = Path("/opt/chexie-knowledge/entities")
RAW_DIRS = {
    1: "/home/ubuntu/workspace/chexie_data_bid1/threads.json",
    2: "/home/ubuntu/workspace/chexie_data_bid2/threads.json",
    3: "/home/ubuntu/workspace/chexie_data_bid3/threads.json",
    4: "/home/ubuntu/workspace/chexie_data_bid4/threads.json",
    7: "/home/ubuntu/workspace/chexie_data_bid7/threads.json",
}

def extract_entities_from_text(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"routes": [], "problems": [], "roles": []}
    for route in ROUTES:
        if route in text:
            found["routes"].append(route)
    for prob in PROBLEMS:
        if prob in text:
            found["problems"].append(prob)
    for role in ROLES:
        if role in text:
            found["roles"].append(role)
    return found

def extract_year_from_title(title: str) -> str | None:
    ys = YEARS_PATTERN.findall(title)
    if ys:
        return ys[0]
    return None

def extract_season_from_title(title: str) -> str | None:
    seasons = BID_PATTERN.findall(title)
    if seasons:
        return seasons[0]
    return None

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # entity → set of tids
    route_tids: dict[str, set[int]] = defaultdict(set)
    problem_tids: dict[str, set[int]] = defaultdict(set)
    role_tids: dict[str, set[int]] = defaultdict(set)
    year_tids: dict[str, set[int]] = defaultdict(set)

    for bid, path in RAW_DIRS.items():
        path = Path(path)
        if not path.exists():
            print(f"  skip bid={bid}: {path} not found")
            continue
        threads = json.loads(path.read_text(encoding="utf-8"))
        for thread in threads:
            tid = int(thread.get("tid") or 0)
            title = str(thread.get("title") or "")
            posts = thread.get("posts") or []

            # Extract year from title
            year = extract_year_from_title(title)
            if year:
                year_tids[year].add(tid)

            # Extract entities from title
            title_entities = extract_entities_from_text(title)
            for route in title_entities["routes"]:
                route_tids[route].add(tid)
            for prob in title_entities["problems"]:
                problem_tids[prob].add(tid)
            for role in title_entities["roles"]:
                role_tids[role].add(tid)

            # Extract entities from all post content
            for post in posts:
                content = str(post.get("content") or "")
                entities = extract_entities_from_text(content)
                for route in entities["routes"]:
                    route_tids[route].add(tid)
                for prob in entities["problems"]:
                    problem_tids[prob].add(tid)
                for role in entities["roles"]:
                    role_tids[role].add(tid)

    # Write output
    def serialize(d: dict[str, set[int]]) -> dict[str, list[int]]:
        return {k: sorted(v) for k, v in d.items()}

    (OUT_DIR / "routes.json").write_text(
        json.dumps(serialize(route_tids), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT_DIR / "problems.json").write_text(
        json.dumps(serialize(problem_tids), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT_DIR / "roles.json").write_text(
        json.dumps(serialize(role_tids), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT_DIR / "years.json").write_text(
        json.dumps(serialize(year_tids), ensure_ascii=False, indent=2),
        encoding="utf-8")

    stats = {
        "routes": {k: len(v) for k, v in sorted(route_tids.items())},
        "problems": {k: len(v) for k, v in sorted(problem_tids.items())},
        "roles": {k: len(v) for k, v in sorted(role_tids.items())},
        "years": {k: len(v) for k, v in sorted(year_tids.items())},
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"routes: {len(route_tids)} terms, {sum(len(v) for v in route_tids.values())} total mappings")
    print(f"problems: {len(problem_tids)} terms, {sum(len(v) for v in problem_tids.values())} total mappings")
    print(f"roles: {len(role_tids)} terms, {sum(len(v) for v in role_tids.values())} total mappings")
    print(f"years: {len(year_tids)} terms, {sum(len(v) for v in year_tids.values())} total mappings")
    print(f"Output: {OUT_DIR}")

if __name__ == "__main__":
    build()
