#!/usr/bin/env python3
"""
Topic analysis engine for chexie-knowledge.
Ties together: retrieval → co-occurrence graph → community detection
             → c-TF-IDF labeling → event mapping → structured output.

Usage:
    from analyze_topic import analyze
    result = analyze("白河双日押后出了什么问题")
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jieba
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
JARGON_PATH = ROOT / "jargon.yaml"
EVENTS_PATH = ROOT / "events.yaml"

# NOTE: events.yaml = activity type schema (拉练/执委会/etc.)
# For controversy event lookups, use scripts/controversy_lookup.py
# which enforces: only candidate tids returned, never post text.
# Controversy index is SECONDARY — always verify facts against original chunks.
# NOTE: events.yaml = activity type schema (拉练/执委会/etc.)
# For controversy event lookups, use scripts/controversy_lookup.py
# which enforces: only candidate tids returned, never post text.
# Controversy index is SECONDARY — always verify facts against original chunks.
ENTITIES_DIR = ROOT / "entities"
PROMPTS_DIR = ROOT / "prompts"

# ── Black话 ──

_JARGONS: dict[str, dict] | None = None


def load_jargons() -> dict:
    global _JARGONS
    if _JARGONS is None:
        if JARGON_PATH.exists():
            with open(JARGON_PATH, encoding="utf-8") as f:
                _JARGONS = yaml.safe_load(f).get("jargons", {})
        else:
            _JARGONS = {}
    return _JARGONS


# ── Event classification ──

_EVENTS: dict | None = None


def load_events() -> dict:
    global _EVENTS
    if _EVENTS is None:
        if EVENTS_PATH.exists():
            with open(EVENTS_PATH, encoding="utf-8") as f:
                _EVENTS = yaml.safe_load(f).get("events", {})
        else:
            _EVENTS = {}
    return _EVENTS


def classify_post_event(title: str, board: str) -> str | None:
    """Classify a single post to event type. Returns event name or None."""
    events = load_events()
    for event_name, config in events.items():
        detect = config.get("detect", {})
        # Check board
        allowed_boards = detect.get("board", [])
        if allowed_boards and board not in allowed_boards:
            continue
        # Check title keywords
        keywords = detect.get("title_keywords", [])
        if any(kw in title for kw in keywords):
            return event_name
    return None


def classify_event_by_majority(posts: list) -> str | None:
    """Classify a group of posts by majority vote. Returns event name or None."""
    votes = Counter()
    for p in posts:
        title = getattr(p, "title", "") if hasattr(p, "title") else p.get("title", "")
        board = getattr(p, "board", "") if hasattr(p, "board") else p.get("board", "")
        event = classify_post_event(str(title), str(board))
        if event:
            votes[event] += 1
    if votes:
        return votes.most_common(1)[0][0]
    return None


# ── c-TF-IDF Topic labeling ──

# Chinese stop words for TF-IDF
STOP_WORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "什么", "怎么", "为什么", "因为", "所以",
    "但是", "如果", "虽然", "而且", "然后", "可以", "可能",
    "这个", "那个", "这些", "那些", "已经", "还是", "或者",
    "不过", "只是", "就是", "不是", "还是", "吗", "吧", "呢",
    "啊", "哦", "嗯", "呀", "么", "qwq", "QAQ", "orz",
    "的", "了", "是", "我", "我们", "你", "你们", "他", "他们",
    "她", "它", "这", "那", "哪", "什么", "怎么", "为什么",
    "一个", "这个", "那个", "这些", "那些", "有些", "所有",
    "已经", "可以", "应该", "需要", "可能", "能够",
    "因为", "所以", "但是", "而且", "然后", "如果", "虽然",
    "没有", "不是", "就是", "还是", "或者", "不过", "只是",
    "吗", "吧", "呢", "啊", "哦", "嗯", "呀", "么", "哈",
    "跟", "对", "把", "被", "让", "给", "替", "为", "从",
    "到", "在", "向", "于", "与", "和", "同", "跟", "把",
    "被", "让", "叫", "给", "比", "跟", "同", "与", "和",
])

_EXTRA_STOP = {
    "然后", "因为", "所以", "但是", "而且", "虽然", "不过",
    "如果", "可以", "应该", "需要", "可能", "就是", "不是",
    "一个", "这个", "那个", "什么", "怎么", "没有", "还是",
    "已经", "时候", "之后", "之前", "现在", "今天", "明天",
    "昨天", "上午", "下午", "晚上", "早上", "中午", "晚上",
    "分钟", "小时", "公里", "km", "速度", "均速",
}
STOP_WORDS.update(_EXTRA_STOP)


def _tokenizer(text: str) -> list[str]:
    """Jieba tokenizer for TfidfVectorizer."""
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in STOP_WORDS]


def label_communities(communities: dict) -> dict[int, str]:
    """c-TF-IDF topic labeling for each community.

    Args:
        communities: {cid: {"label": ..., "posts": [PostNode, ...]}}

    Returns:
        {cid: "话题标签（如 白河·押后·扎胎）"}
    """
    cids = list(communities.keys())

    if len(communities) == 0:
        return {}

    # Build corpus: one doc per community
    docs = []
    for cid in cids:
        text_parts = []
        for p in communities[cid]["posts"]:
            title = getattr(p, "title", "")
            text = getattr(p, "text", "")
            text_parts.append(f"{title} {text[:500]}")
        docs.append(" ".join(text_parts))

    if len(cids) == 1:
        # Single community: take top keywords
        tokens = _tokenizer(docs[0])
        freq = Counter(tokens)
        top = [w for w, _ in freq.most_common(10) if len(w) > 1][:3]
        return {cids[0]: "·".join(top) if top else "综合讨论"}

    try:
        vec = TfidfVectorizer(
            max_features=2000,
            tokenizer=_tokenizer,
            max_df=0.85,
            min_df=1,
        )
        tfidf = vec.fit_transform(docs)
        feature_names = vec.get_feature_names_out()
    except ValueError:
        # All docs empty or all same tokens
        return {cid: "综合讨论" for cid in cids}

    labels = {}
    for i, cid in enumerate(cids):
        row = tfidf[i].toarray()[0]
        top3_idx = row.argsort()[-3:][::-1]
        top_words = [feature_names[j] for j in top3_idx if row[j] > 0.05]
        top_words = [w for w in top_words if w not in STOP_WORDS][:3]
        labels[cid] = "·".join(top_words) if top_words else "综合讨论"

    return labels


# ── Jargon annotation ──

def annotate_jargons(text: str) -> list[dict]:
    """Find jargon terms in text and return their explanations.

    Returns [{"term": "押后", "explain": "随行修车员..."}, ...]
    """
    jargons = load_jargons()
    found = []
    for term, config in jargons.items():
        if term in text:
            found.append({
                "term": term,
                "explain": config.get("explain", ""),
            })
    return found


# ── Output formatting ──

def format_analysis(
    query: str,
    communities: dict,
    labels: dict[int, str],
    events: dict[int, str | None],
    board_weights: dict[str, float],
) -> str:
    """Generate the final structured analysis text."""
    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"【专题分析】{query}")
    lines.append("")

    total_posts = sum(len(c["posts"]) for c in communities.values())
    lines.append(f"共 {len(communities)} 个话题、{total_posts} 个帖子")
    lines.append("")

    for cid, community in communities.items():
        label = labels.get(cid, "未分类")
        posts = community["posts"]
        max_weight = community["max_weight"]
        stars = "★" * min(5, max(1, round(max_weight)))
        event_type = events.get(cid)

        lines.append(f"── 话题 {cid + 1}：{label} ──")
        lines.append(f"  包含 {len(posts)} 个帖子（权重: {stars}）")
        if event_type:
            lines.append(f"  事件类型: {event_type}")
        lines.append("")

        for p in posts:
            bw_stars = "★" * min(5, max(1, round(p.board_weight)))
            lines.append(f"  [{bw_stars}]【{p.board}】{p.title}")
            if p.author and p.date:
                lines.append(f"    作者: {p.author} | 日期: {p.date[:10]}")
            if p.url:
                lines.append(f"    → {p.url}")
            # Show first 200 chars of text
            text_preview = p.text[:200].replace("\n", " ").strip()
            if text_preview:
                lines.append(f"    {text_preview}...")
            lines.append("")

    # Jargon annotations
    all_text = " ".join(
        p.title + " " + p.text[:200]
        for c in communities.values()
        for p in c["posts"]
    )
    jargons_found = annotate_jargons(all_text)
    if jargons_found:
        lines.append(f"── 黑话注释 ──")
        for j in jargons_found[:8]:
            lines.append(f"  · {j['term']}：{j['explain']}")
        lines.append("")

    # Source confidence note
    lines.append(f"── 信源说明 ──")
    lines.append(f"  ★★★★★ 工作区（正式通知/执委会记录/队长总结）= 协会正式决定")
    lines.append(f"  ★★★★  行者足音（个人总结/经验分享）= 个人视角，需交叉验证")
    lines.append(f"  ★★★   一技之长/车友宝典（技术经验）= 通用技术参考")
    lines.append(f"  ★★    纯净水（闲聊）= 仅作补充参考")

    return "\n".join(lines)


# ── Main entry point ──

def analyze(
    query: str,
    retrieved_posts: list[dict],
    board_weights: dict[str, float],
) -> dict[str, Any]:
    """Run full analysis pipeline.

    Args:
        query: user query
        retrieved_posts: list of post dicts from pipeline (with tid, title, board, text, ...)
        board_weights: {board_name: weight} dict

    Returns:
        {
            "report": str,          # formatted analysis text
            "communities": {...},   # raw community data
            "labels": {...},        # c-TF-IDF labels
            "events": {...},        # classified event types
            "jargons": [...],       # jargon annotations
        }
    """
    from scripts.cooccur_graph import cluster_posts, PostNode

    # Step 1: Build co-occurrence graph + detect communities
    clustering = cluster_posts(retrieved_posts, board_weights)
    raw_communities = clustering["communities"]

    # Step 2: c-TF-IDF labeling
    labels = label_communities(raw_communities)

    # Step 3: Event classification
    events = {}
    for cid, community in raw_communities.items():
        event_type = classify_event_by_majority(community["posts"])
        events[cid] = event_type

    # Step 4: Format output
    report = format_analysis(query, raw_communities, labels, events, board_weights)

    # Step 5: Jargon annotations
    all_text = " ".join(
        p.title + " " + p.text[:300]
        for c in raw_communities.values()
        for p in c["posts"]
    )
    jargons_found = annotate_jargons(all_text)

    return {
        "report": report,
        "communities": {
            cid: {
                "label": labels.get(cid, ""),
                "posts": [
                    {"tid": p.tid, "title": p.title, "board": p.board,
                     "author": p.author, "date": p.date, "url": p.url}
                    for p in c["posts"]
                ],
                "max_weight": c["max_weight"],
                "event_type": events.get(cid),
            }
            for cid, c in raw_communities.items()
        },
        "labels": labels,
        "events": events,
        "jargons": jargons_found,
    }
