#!/usr/bin/env python3
"""
Lightweight co-occurrence graph + community detection for chexie-knowledge.

Builds a graph where nodes = posts, edges = shared entities (routes, problems, roles).
Runs Louvain community detection to automatically group related posts.

Usage (programmatic):
    from cooccur_graph import cluster_posts
    communities = cluster_posts(posts, entities_index)

This module is designed to run on a query result subset (50-100 posts),
not on the entire corpus.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
from community import community_louvain

ROOT = Path(__file__).resolve().parent.parent
ENTITIES_DIR = ROOT / "entities"


# ── Entity index loading ──

_ENTITY_CACHE: dict[str, dict[str, list[int]]] | None = None


def load_entities() -> dict[str, dict[str, list[int]]]:
    """Load entity mappings (route/problem/role → list of tids)."""
    global _ENTITY_CACHE
    if _ENTITY_CACHE is not None:
        return _ENTITY_CACHE

    entities: dict[str, dict[str, list[int]]] = {}
    for name in ("routes", "problems", "roles", "years"):
        path = ENTITIES_DIR / f"{name}.json"
        if path.exists():
            entities[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            entities[name] = {}
    _ENTITY_CACHE = entities
    return entities


def get_entity_labels(text: str, title: str, entities: dict) -> dict[str, list[str]]:
    """Extract entity labels from a post's text + title.

    Returns {"routes": [...], "problems": [...], "roles": [...]}
    """
    labels: dict[str, list[str]] = {"routes": [], "problems": [], "roles": []}
    haystack = f"{title}\n{text}"

    for category in ("routes", "problems", "roles"):
        for entity_name, tids in entities.get(category, {}).items():
            if entity_name in haystack:
                labels[category].append(entity_name)

    return labels


# ── Post Node ──

class PostNode:
    """A node in the co-occurrence graph, representing one unique post."""

    def __init__(self, tid: str, title: str, board: str, board_weight: float,
                 text: str = "", entities: dict[str, list[str]] | None = None,
                 url: str = "", author: str = "", date: str = ""):
        self.tid = tid
        self.title = title
        self.board = board
        self.board_weight = board_weight
        self.text = text
        self.entities = entities or {"routes": [], "problems": [], "roles": []}
        self.url = url
        self.author = author
        self.date = date

    def __repr__(self):
        return f"<PostNode {self.tid}: {self.title[:30]}>"


# ── Graph building ──

def build_cooccur_graph(posts: list[PostNode]) -> nx.Graph:
    """Build weighted co-occurrence graph from a list of PostNodes.

    Edge weight calculation:
      - Shared route: +2 per shared route
      - Shared problem: +2 per shared problem
      - Shared role: +1 per shared role
      - Same board: +0.5
      - Same year (from title): +1

    Returns NetworkX Graph with node attributes and weighted edges.
    """
    import re

    G = nx.Graph()

    for p in posts:
        G.add_node(p.tid, post=p)

    if len(posts) < 2:
        return G

    # Extract year from each post title
    def extract_year(title: str) -> str | None:
        m = re.search(r'(?:20)?(\d{2})(?=春|秋|暑|冬|双日|单日)', title)
        return m.group(0) if m else None

    years = {p.tid: extract_year(p.title) for p in posts}

    for i, a in enumerate(posts):
        for b in posts[i + 1:]:
            w = 0

            # Shared routes
            shared_routes = set(a.entities.get("routes", [])) & set(b.entities.get("routes", []))
            w += len(shared_routes) * 2

            # Shared problems
            shared_probs = set(a.entities.get("problems", [])) & set(b.entities.get("problems", []))
            w += len(shared_probs) * 2

            # Shared roles
            shared_roles = set(a.entities.get("roles", [])) & set(b.entities.get("roles", []))
            w += len(shared_roles) * 1

            # Same board
            if a.board == b.board:
                w += 0.5

            # Same year
            if years[a.tid] and years[b.tid] and years[a.tid] == years[b.tid]:
                w += 1

            if w > 0:
                G.add_edge(a.tid, b.tid, weight=w)

    return G


# ── Community detection ──

def detect_communities(G: nx.Graph) -> dict[int, list[str]]:
    """Run Louvain community detection on the graph.

    Returns {community_id: [tid1, tid2, ...]}
    Nodes with no edges are grouped as "singletons" (community_id=-1).
    """
    if G.number_of_nodes() == 0:
        return {}

    if G.number_of_edges() == 0:
        # No connections: each node is its own community
        return {-1: list(G.nodes())}

    partition = community_louvain.best_partition(G, weight="weight")

    communities: dict[int, list[str]] = defaultdict(list)
    for tid, cid in partition.items():
        communities[cid].append(tid)

    return dict(communities)


# ── Orphan handling ──

def find_orphans(posts: list[PostNode], G: nx.Graph) -> list[PostNode]:
    """Find posts that have no edges (no shared entities with any other post)."""
    orphans = []
    for p in posts:
        if p.tid not in G or G.degree(p.tid) == 0:
            orphans.append(p)
    return orphans


# ── Main entry point ──

def cluster_posts(posts: list[dict], board_weights: dict[str, float]) -> dict:
    """Full pipeline: build graph → detect communities → return grouped results.

    Args:
        posts: list of search result dicts with keys:
               tid, title, board, text, author, date, url (each from pipeline)
        board_weights: {board_name: weight} dict

    Returns:
        {
            "communities": {cid: {"label": ..., "posts": [PostNode, ...]}},
            "orphans": [PostNode, ...],
            "graph": nx.Graph,
        }
    """
    entities = load_entities()

    # Build PostNodes, extracting entities
    node_posts = []
    for p in posts:
        source = p.get("info") or p.get("source") or {}
        tid = str(source.get("tid", p.get("tid", "")))
        title = str(source.get("title", p.get("title", "")))
        board = str(source.get("board", p.get("board", "")))
        text = str(p.get("text", ""))
        author = str(source.get("author", p.get("author", "")))
        date = str(source.get("time", p.get("date", "")))
        url = str(source.get("url", p.get("url", "")))

        bw = board_weights.get(board, 1.0)
        post_entities = get_entity_labels(text, title, entities)

        node_posts.append(PostNode(
            tid=tid, title=title, board=board, board_weight=bw,
            text=text, entities=post_entities,
            url=url, author=author, date=date,
        ))

    # Build graph
    G = build_cooccur_graph(node_posts)

    # Detect communities
    raw_communities = detect_communities(G)

    # Organize results
    tid_to_post = {p.tid: p for p in node_posts}

    result_communities: dict = {}
    for cid, tids in raw_communities.items():
        community_posts = [tid_to_post[tid] for tid in tids if tid in tid_to_post]
        # Use highest board weight post's title as initial label
        community_posts.sort(key=lambda p: p.board_weight, reverse=True)
        result_communities[cid] = {
            "label": "",  # filled in by c-TF-IDF later
            "posts": community_posts,
            "max_weight": max(p.board_weight for p in community_posts) if community_posts else 1.0,
        }

    # Find orphans (posts not in any community or with no edges)
    orphans = find_orphans(node_posts, G)

    return {
        "communities": result_communities,
        "orphans": orphans,
        "graph": G,
        "node_count": len(node_posts),
        "community_count": len(raw_communities),
    }
