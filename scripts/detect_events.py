#!/usr/bin/env python3
"""
车协知识库事件自动检测脚本 v2

从 FTS5 索引中自动检测历史争议事件，输出结构化事件索引。

多维度评分算法：
1. 被引次数：追踪 "引用自 XXX：" 显式引用标记
2. 作者权威分：理事会/主席团等集体账号权重
3. 突发密度：短时间内回复密度异常
4. 标题信号词：争议/讨论/改革等关键词
5. 回复数：高回复帖子优先

使用方式：
  python detect_events.py                    # 全量扫描
  python detect_events.py --bid 1            # 只扫描车协工作区
  python detect_events.py --min-score 15     # 只输出评分>=15的帖子
  python detect_events.py --output json      # JSON格式输出

输出：候选事件帖子列表，按评分降序排列。
"""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# === 配置 ===

DB_PATH = "/opt/chexie-knowledge/chexie_fts.db"

# 作者权威分映射
AUTHORITY_DICT = {
    '理事会': 10,
    '主席团': 8,
    '理事长': 8,
    '文体部': 6,
    '实践部': 6,
    '执委会': 5,
    '宣传部': 4,
    '外联部': 4,
    '组织部': 4,
    '队医组': 3,
    '技术组': 3,
}

# 标题信号词
TITLE_SIGNALS = [
    # 争议性
    '争议', '讨论', '质疑', '反对', '抗议', '不满', '批评',
    # 改革性
    '改革', '调整', '变更', '修订', '修改', '取消',
    # 事件性
    '事件', '事故', '问题', '冲突', '矛盾',
    # 制度性
    '制度', '规则', '条例', '规定', '办法',
]

# 内容信号词（首楼）
CONTENT_SIGNALS = [
    '不告而取', '工作失误', '违规', '作弊', '渎职',
    '弹劾', '处分', '停权', '开除', '永不录用',
    '取消资格', '警告', '锁帖', '删帖', '禁言',
]

# 排除的标题关键词（例行公告）
EXCLUDE_TITLE = [
    '报名', '拉练通知', '预报名', '团购', '征集暑期',
    '暑期日志', '队医日志', '押后日志', '生日快乐',
    '生日帖', '暑期日记', '足音塔楼', '发车顺序',
    'Can\'t see', 'SaySomething', 'test', '咆哮',
    '心态崩', '开心', '广播', '队长总结', '负责总结',
    '执委会记录', '执委会通知', '主席团征集', '物资',
    '免检特检', '主任任命', '执委会任命', '理事会关于20',
    '理事会征集20', '理事会关于任命', '【任命】',
    '考挂标准', '技术组考核', '训练通知', '体测通知',
    '前站招募', '追离队', '药品管理',
]

# 排除的版面（纯技术讨论）
# 注意：bid=3（车友宝典）和bid=7（一技之长）不排除，但通过标题过滤
# 因为这些版面偶尔也有制度性讨论


def load_posts(bid_filter=None):
    """从 FTS5 数据库加载所有帖子切片"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    if bid_filter:
        cur.execute(
            "SELECT chunk_id, title, board, author, text FROM fts5_index WHERE chunk_id LIKE ?",
            (f'bid{bid_filter}_%',)
        )
    else:
        cur.execute("SELECT chunk_id, title, board, author, text FROM fts5_index")
    
    rows = cur.fetchall()
    conn.close()
    return rows


def build_post_index(rows):
    """构建帖子索引和作者映射"""
    post_info = {}
    post_authors = defaultdict(set)
    
    for chunk_id, title, board, author, text in rows:
        bt = re.search(r'bid(\d+)_tid(\d+)', chunk_id)
        if not bt:
            continue
        
        bid, tid = bt.group(1), bt.group(2)
        key = f"{bid}_{tid}"
        
        # 提取日期
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text[:200])
        if not date_match:
            continue
        date = date_match.group(1)
        
        # 提取楼层
        floor_match = re.search(r'第(\d+)楼', text[:100])
        floor = floor_match.group(1) if floor_match else '?'
        
        # 初始化帖子信息
        if key not in post_info:
            post_info[key] = {
                'bid': bid,
                'tid': tid,
                'title': title,
                'board': board,
                'dates': [],
                'count': 0,
                'first_floor': '',
            }
        
        info = post_info[key]
        info['count'] += 1
        info['dates'].append(date)
        post_authors[key].add(author)
        
        # 保存首楼内容
        if floor in ('0', '1') and not info['first_floor']:
            content = re.sub(r'【.*?】.*?第\d+楼.*?链接.*?\n', '', text[:600], count=1)
            info['first_floor'] = content[:400]
    
    return post_info, post_authors


def build_quote_graph(rows, post_authors):
    """构建引用图谱（追踪显式引用标记）"""
    quote_graph = defaultdict(lambda: {'cited_by': set(), 'cites': []})
    
    for chunk_id, title, board, author, text in rows:
        bt = re.search(r'bid(\d+)_tid(\d+)', chunk_id)
        if not bt:
            continue
        
        citing_key = f"{bt.group(1)}_{bt.group(2)}"
        
        # 提取显式引用标记：引用自 XXX：
        matches = re.findall(r'引用自\s*(.+?)\s*：', text[:300])
        for quoted_author in matches:
            quoted_author = quoted_author.strip()
            
            # 查找该作者发过的帖子
            for target_key, authors in post_authors.items():
                if target_key != citing_key and quoted_author in authors:
                    quote_graph[citing_key]['cites'].append(quoted_author)
                    quote_graph[target_key]['cited_by'].add(citing_key)
    
    return quote_graph


def calculate_authority(authors):
    """计算帖子的作者权威分（取最高）"""
    max_auth = 0
    for author in authors:
        for key, score in AUTHORITY_DICT.items():
            if key in author:
                max_auth = max(max_auth, score)
    return min(max_auth, 10)


def detect_burst(dates):
    """检测突发密度（1天内>=5回复 或 3天内>=8回复）"""
    if len(dates) < 5:
        return False
    
    sorted_dates = sorted(dates)
    
    # 1天内>=5回复
    for i in range(len(sorted_dates) - 4):
        d1 = datetime.strptime(sorted_dates[i], '%Y-%m-%d')
        d2 = datetime.strptime(sorted_dates[i + 4], '%Y-%m-%d')
        if (d2 - d1).days <= 1:
            return True
    
    # 3天内>=8回复
    if len(dates) >= 8:
        for i in range(len(sorted_dates) - 7):
            d1 = datetime.strptime(sorted_dates[i], '%Y-%m-%d')
            d2 = datetime.strptime(sorted_dates[i + 7], '%Y-%m-%d')
            if (d2 - d1).days <= 3:
                return True
    
    return False


def calculate_title_signals(title):
    """计算标题信号词数量"""
    count = 0
    for signal in TITLE_SIGNALS:
        if signal in title:
            count += 1
    return count


def calculate_content_signals(first_floor):
    """计算首楼内容信号词数量"""
    count = 0
    for signal in CONTENT_SIGNALS:
        if signal in first_floor:
            count += 1
    return count


def should_exclude(title):
    """判断是否应排除（例行公告）"""
    for exclude in EXCLUDE_TITLE:
        if exclude in title:
            return True
    return False


def score_posts(post_info, post_authors, quote_graph, min_score=0):
    """多维度评分"""
    results = []
    
    for key, info in post_info.items():
        # 基础过滤
        if info['count'] < 5:
            continue
        
        if should_exclude(info['title']):
            continue
        
        # 维度1：被引次数（上限20分）
        cited = len(quote_graph[key]['cited_by'])
        cited_score = min(cited * 2, 20)
        
        # 维度2：作者权威（上限10分）
        auth_score = calculate_authority(post_authors[key])
        
        # 维度3：突发密度（10分或0分）
        is_burst = detect_burst(info['dates'])
        burst_score = 10 if is_burst else 0
        
        # 维度4：标题信号词（每个3分）
        title_sig = calculate_title_signals(info['title'])
        title_score = title_sig * 3
        
        # 维度5：回复数（每5回复1分，上限10分）
        reply_score = min(info['count'] // 5, 10)
        
        # 维度6：首楼内容信号词（每个5分，上限15分）
        content_sig = calculate_content_signals(info['first_floor'])
        content_score = min(content_sig * 5, 15)
        
        # 总分
        total = cited_score + auth_score + burst_score + title_score + reply_score + content_score
        
        if total >= min_score:
            results.append({
                'bid': info['bid'],
                'tid': info['tid'],
                'title': info['title'],
                'board': info['board'],
                'score': total,
                'count': info['count'],
                'dates': sorted(info['dates']),
                'cited': cited,
                'auth': auth_score,
                'burst': is_burst,
                'title_sig': title_sig,
                'content_sig': content_sig,
                'first_floor': info['first_floor'][:150],
            })
    
    # 按评分降序排列
    results.sort(key=lambda x: -x['score'])
    return results


def main():
    parser = argparse.ArgumentParser(description='车协知识库事件自动检测')
    parser.add_argument('--bid', type=int, help='只扫描指定版面（1=车协工作区, 2=行者足音, 3=车友宝典, 4=纯净水, 7=一技之长）')
    parser.add_argument('--min-score', type=int, default=0, help='最低评分阈值')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--limit', type=int, default=50, help='输出数量限制')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载数据（bid={args.bid or '全部'}）...", flush=True)
    rows = load_posts(args.bid)
    print(f"加载了 {len(rows)} 条切片", flush=True)
    
    # 构建索引
    print("构建帖子索引...", flush=True)
    post_info, post_authors = build_post_index(rows)
    print(f"索引了 {len(post_info)} 个帖子", flush=True)
    
    # 构建引用图谱
    print("构建引用图谱...", flush=True)
    quote_graph = build_quote_graph(rows, post_authors)
    print(f"追踪了 {len(quote_graph)} 个引用关系", flush=True)
    
    # 评分
    print("多维度评分...", flush=True)
    results = score_posts(post_info, post_authors, quote_graph, args.min_score)
    print(f"评分完成，{len(results)} 个候选事件", flush=True)
    
    # 输出
    if args.output == 'json':
        output = results[:args.limit]
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n=== 候选事件（Top {args.limit}）===\n")
        for i, r in enumerate(results[:args.limit], 1):
            print(f"{i}. [{r['bid']}] tid={r['tid']} | 评分={r['score']}")
            print(f"   标题：{r['title']}")
            print(f"   维度：回复={r['count']} | 被引={r['cited']} | 权威={r['auth']} | 突发={r['burst']} | 标题信号={r['title_sig']} | 内容信号={r['content_sig']}")
            print(f"   时间：{r['dates'][0]} ~ {r['dates'][-1]}")
            if r['first_floor']:
                print(f"   首楼：{r['first_floor']}...")
            print()


if __name__ == '__main__':
    main()
