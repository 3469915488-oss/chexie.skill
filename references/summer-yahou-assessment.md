# 暑期团押后能力四维评估方法论

## 评估框架

四个维度：
1. **押后考核通过时间**（相对于春季跑步体测）——区分有/无拉练实操经验
2. **拉练担任押后次数**——实战经验量化
3. **押后负责/冬游押后负责经历**——技术领导力
4. **技术组考核参与/通过**——技术天花板

## 数据源速查表

| 维度 | 数据源 | 版面 | 获取方式 |
|------|--------|------|----------|
| 押后考核通过名单+时间 | 实践部年度押后考挂标准帖 | 车友宝典 bid=3 | FAISS 搜索 "XX年度 押后 新增 名单" |
| 拉练押后次数 | **押后日志帖** | 一技之长 bid=7 | 从 faiss_meta.jsonl 提取全部 chunks，解析每条记录 |
| 押后负责任命 | 实践部任命帖 | 车协工作区 bid=1 | FAISS 搜索 "XX暑期 押后负责 任命" |
| 冬游押后负责 | 冬游相关帖 | 车协工作区 bid=1 | FAISS 搜索 "XX冬游 押后负责" |
| 技术组通过名单 | 实践部技术组考核帖 | 车协工作区 bid=1 | FAISS 搜索 "XX技术组 考核 通过名单" |
| 技术组报名/单项优秀 | 同上帖子 | 车协工作区 bid=1 | 同一帖子后续楼层 |
| 暑期团押后成员 | **暑期押后日志帖**首楼 | 一技之长 bid=7 | FAISS 搜索 "XX实践团 暑期押后日志" |
| 暑期预备队员名单 | 理事会公告 | 车协工作区 bid=1 | FAISS 搜索 "XX暑期 预备队员 名单" |
| 跑步体测时间 | 文体部体测通知 | 车协工作区 bid=1 | FAISS 搜索 "XX春训 跑步体测 通知" |

## 已知帖子 tid 索引

### 押后日志帖（拉练押后次数）
- 22-23 学年：tid=1009
- 23-24 学年：tid=1058
- 24-25 学年：tid=1119

### 暑期押后日志帖（团内押后名单）
- 24 实践团：tid=1116
- 24 骑行团：tid=1109
- 24 飞行团：tid=1113
- 24 轻骑团：tid=1114
- 25 实践团：tid=1173
- 25 骑行团：tid=1176

### 押后考核通过名单
- 23-24 年度：tid=4966（发布 2023-09-11）
- 24-25 年度：tid=5043（发布 2024-09-13）
- 25-26 年度：待确认

### 押后负责任命
- 2024：tid=9912（2024-06-02）
- 2025：tid=10299（2025-06-01）
- 2026：未发布

### 技术组考核
- 2024：tid=9900（无人通过，单项优秀名单在后续楼层）
- 2025：tid=10285（烟火通过，单项优秀名单在后续楼层）
- 2026：tid=10672（crashed/墨/树懒2号通过），报名帖 tid=10637

### 暑期预备队员名单
- 2024：tid=9904
- 2025：tid=10292
- 2026：tid=10687

### 跑步体测时间
- 2024 春训：tid=9867（5/8-5/10）
- 2025 春训：tid=10267（5/9-5/10）
- 2026 春训：tid=10652（5/18-5/20）

## 提取代码模板

### 从 FAISS meta 提取押后日志记录

```python
import json, re
from collections import defaultdict

target_tids = {1058: "23-24", 1119: "24-25"}
all_entries = defaultdict(list)

with open('/opt/chexie-knowledge/faiss_meta.jsonl', 'r') as f:
    for line in f:
        row = json.loads(line)
        src = row.get('source', {})
        if not isinstance(src, dict):
            continue
        tid = src.get('tid')
        if tid not in target_tids:
            continue
        
        text = row.get('text', '')
        author = src.get('author', '')
        year_label = target_tids[tid]
        
        # 解析拉练名称、职务、时间
        lalian_m = re.search(r'(?:拉练名称|名称|【名称】)[：:]\s*(.+?)(?:\n|$)', text)
        zhiwu_m = re.search(r'(?:职务|【职务】)[：:]\s*(.+?)(?:\n|$)', text)
        time_m = re.search(r'(?:拉练时间|时间|【时间】)[：:]\s*(.+?)(?:\n|$)', text)
        
        lalian_name = lalian_m.group(1).strip() if lalian_m else '?'
        zhiwu = zhiwu_m.group(1).strip() if zhiwu_m else '押后'
        entry_time = time_m.group(1).strip() if time_m else src.get('time', '')[:10]
        
        if '押后' in text and author and author not in ['实践部', 'CAPU']:
            all_entries[year_label].append({
                'author': author,
                'lalian': lalian_name,
                'zhiwu': zhiwu,
                'time': entry_time,
            })

# 按作者聚合统计
counts = defaultdict(list)
for e in all_entries['24-25']:  # 选择学年
    if e['lalian'] != '?':
        counts[e['author']].append(e)

for author, entries in sorted(counts.items(), key=lambda x: -len(x[1])):
    unique = len(set(e['lalian'] for e in entries))
    print(f"{author}: {len(entries)}次押后, {unique}个不同拉练")
```

## 注意事项

1. **押后日志是自报告的**——不是每个担任押后的人都会回帖写日志。日志中的次数是**下限**，不是上限。
2. **暑期押后日志首楼的押后名单是出发时的状态**——可能有人在暑期路上新考过押后（以日志正文中为准）。
3. **体测时间 vs 押后通过时间**：押后考核是持续进行的，但 24-25 年度的押后名单帖子发布于 9 月，之后在春季学期会通过回帖追加新增押后。判断"体测前/后"需要查回帖日期。
4. **"nb闪闪的押后"**（橙色标注）在早期帖子中是技术水平较高的标志。
5. **2026 暑期押后负责尚未任命**（截至 2026-06-08），技术组 3 人（crashed/墨/树懒2号）是潜在人选。
