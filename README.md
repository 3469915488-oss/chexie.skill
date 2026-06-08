# 车协知识库 · Chexie Knowledge

> **北大车协论坛知识库**——检索车协工作区、行者足音、纯净水、车友宝典、一技之长五大版块的 13.8 万条帖子切片，基于有来源、可回查的论坛记录做搜索和专题分析。

---

## 它能做什么

### 精准搜索

```bash
python scripts/search_chexie.py "白河 押后 扎胎" --top-k 5
```

快速命中相关帖子，每条带版面名、标题、楼层、作者、原文链接。

### 专题分析

```bash
python scripts/search_chexie.py "体测标准为什么改了" --analyze
```

自动聚合相关帖子 → 共现图社区发现 → 话题分组 → c-TF-IDF 自动标签 → 事件类型识别 → 结构化分析输出。

### 深度复盘

```bash
python scripts/search_chexie.py --pipeline "25双日白河A组出现了什么问题"
```

三层输出：目标拉练全貌（问题+方案+反思）→ 同期对照 → 往年经验。

---

## 架构

### 三路检索 → RRF 融合

```
用户查询
  │
  ├─ FAISS 内容索引（138001 条切片，BGE-small-zh 512维）
  ├─ FAISS 标题索引（19451 条帖子标题，权重 2x）
  └─ SQLite FTS5 BM25（零内存占用，含黑话扩展）
  │
  ▼
  RRF 三路融合 → 精排 Top-K
```

- **模型**：BAAI/bge-small-zh-v1.5（512 维，CPU 高效推理）
- **内容索引**：FAISS IndexFlatIP，13.8 万条
- **标题索引**：独立 FAISS，搜索时加权 2x
- **BM25**：SQLite FTS5，零内存占用，比 rank_bm25 快 100 倍

### 动态路由

| 查询特征 | 路由策略 |
|----------|---------|
| 含地名/数字/制度术语（"白河""2025""执委会"） | BM25 权重提升，精确命中 |
| 经验/历史/争议（"怎么回事""怎么处理"） | 向量权重提升，语义模糊搜索 |
| 混合（"25双日白河押后出了什么问题"） | 均衡融合 + 全帖检索 |

### 专题分析引擎

```
搜索命中帖子集
  │
  ├─ 共现图建图（节点=帖子，边=共享实体）
  ├─ Louvain 社区发现 → 自动话题分组
  ├─ c-TF-IDF → 自动话题标签
  ├─ 事件框架映射（拉练/执委会/出摊等）
  ├─ 黑话自动注释
  └─ 按指令模板输出结构化分析
```

---

## 数据源

| 版面 | bid | 切片数 | 权重 | 用途 |
|------|-----|--------|------|------|
| 车协工作区 | 1 | 44,434 | ★★★★★ | 执委会记录、正式通知、队长总结 |
| 行者足音 | 2 | 55,718 | ★★★★ | 个人骑行经验、游记、讨论 |
| 一技之长 | 7 | 8,950 | ★★★ | 通用技术经验 |
| 车友宝典 | 3 | 8,802 | ★★ | 装备知识、操作指南 |
| 纯净水 | 4 | 20,097 | ★ | 闲聊参考 |

总计 **138,001 条检索切片**，覆盖 2003-2026 年约 3.1 万篇帖子。

### 时间跨度

最早帖子：2003 年（一技之长），最新帖子：2026 年春。

---

## 快速使用

```bash
# 快速检索（默认）
python scripts/search_chexie.py "远征报名流程" --top-k 8

# 专题分析（自动聚类）
python scripts/search_chexie.py "体测标准为什么改了" --analyze

# 深度复盘
python scripts/search_chexie.py --pipeline "25双日白河A组出现了什么问题"

# JSON 输出
python scripts/search_chexie.py "春训时间" --top-k 5 --json
```

### 全部参数

| 参数 | 说明 |
|------|------|
| `query` | 检索问题 |
| `--top-k N` | 返回结果数（默认 8） |
| `--pipeline` / `-p` | 深度复盘管线（全帖检索 + 内容过滤） |
| `--analyze` / `-a` | 专题分析模式（话题聚类 + c-TF-IDF + 事件映射） |
| `--json` | JSON 格式输出 |
| `--info` | 查看索引状态 |
| `--bm25` | 启用 BM25 混合搜索（默认开启） |
| `--verbose` / `-v` | 详细输出 |

---

## 配置

### 黑话映射（`jargon.yaml`）

车协有大量内部术语——押后、牵车、留口、拿龙、回炉、出摊等。黑话映射表记录了每条术语的标准解释和搜索扩展词，搜索时自动展开。

### 事件框架（`events.yaml`）

定义了五种常见事件类型（拉练、执委会、出摊、拉练报名、修车讲座），每个事件有固定结构（路线/年份/组别、会议届次、角色、常见问题等）。分析模式自动将帖子归类到事件框架中。

### 分析 Prompt（`prompts/*.yaml`）

三段可配置的 LLM prompt 模板，控制分析输出的风格和深度。支持默认分析、争议深入分析、简洁总结三种模式。

### 结构化事件索引（`events/*.yaml`）

14 个历史争议事件的结构化索引（2011-2026），每个事件包含：
- **概述**：事件背景与核心争议
- **阶段时间线**：事件发展的关键节点
- **核心帖子**：相关论坛帖子链接（tid、标题、URL）
- **关键词**：搜索与分类用
- **关键人物**：事件参与者及其立场
- **各方立场**：支持/反对/中间三方的观点
- **关联事件**：与其他事件的联系

**事件列表**：
1. 2011-版主争议事件
2. 2013-纪律争论事件
3. 2013-甘道夫封禁控诉事件
4. 2016-金万琳作弊处罚事件
5. 2017-主席团批评文体部长事件
6. 2018-取消师徒制事件
7. 2018-破风致歉事件
8. 2024-反对 bbzb 事件
9. 2025-暑期分团事件
10. 2026-劳模制度批评帖事件
11. 2026-模式口拉练冲突事件
12. 2026-押后拆夹器事件
13. 2026-冬游特批事件
14. 2026-冬游C组押后任命事件

这些索引用于快速定位历史争议事件的完整脉络，避免重复检索。

---

## 安装

### 依赖

```
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
numpy>=1.21.0
networkx>=2.8
python-louvain>=0.16
scikit-learn>=1.0
jieba>=0.42.1
pyyaml>=6.0
```

模型 `BAAI/bge-small-zh-v1.5` 首次运行自动下载（～200MB）。

### 下载数据

```
wget https://github.com/3469915488-oss/chexie.skill/releases/download/v3.0.0/chexie_data.tar.gz
tar -xzf chexie_data.tar.gz
```

数据文件：faiss_index.bin（270MB）、title_index.bin（38MB）、chexie_fts.db（343MB）、faiss_meta.jsonl（228MB）、entities/\*.json。

---

## 目录结构

```
chexie.skill/
├── SKILL.md                  # Hermes Agent Skill 定义
├── README.md                 # 本文件
├── AGENTS.md                 # AI Agent 配置
├── jargon.yaml               # 黑话映射表（35条）
├── events.yaml               # 事件框架（5种）
├── prompts/
│   ├── analyze_default.yaml      # 默认分析 prompt
│   ├── analyze_controversy.yaml  # 争议深入分析 prompt
│   └── summarize_topic.yaml     # 简洁总结 prompt
├── events/                   # 结构化事件索引（14个）
│   ├── 2011-moderator-dispute.yaml
│   ├── 2013-discipline-debate.yaml
│   ├── 2013-gandalf-ban-protest.yaml
│   ├── 2016-spring-training-cheating.yaml
│   ├── 2017-chairman-criticizes-minister.yaml
│   ├── 2018-cancel-apprenticeship.yaml
│   ├── 2018-pofeng-apology.yaml
│   ├── 2024-anti-bbzb.yaml
│   ├── 2025-team-split.yaml
│   ├── 2026-laomo-institutional-critique.yaml
│   ├── 2026-moshikou-conflict.yaml
│   ├── 2026-rear-guard-parts-incident.yaml
│   ├── 2026-winter-trip-exception.yaml
│   └── 2026-winter-trip-rear-guard.yaml
├── requirements.txt          # Python 依赖
├── scripts/
│   ├── search_chexie.py      # 主入口（搜索 + 管线 + 分析）
│   ├── pipeline.py           # 深度复盘管线
│   ├── build_chexie_faiss.py # FAISS 索引构建（含标题索引）
│   ├── build_entities.py     # 实体映射构建
│   ├── build_fts5.py         # SQLite FTS5 全文索引构建
│   ├── build_bm25.py         # BM25 索引构建（旧版，新系统用 FTS5）
│   ├── build_title_index.py  # 标题索引独立构建
│   ├── search_fts5.py        # FTS5 搜索模块
│   ├── cooccur_graph.py      # 共现图 + 社区发现
│   ├── analyze_topic.py      # 话题分析引擎
│   └── server_search_chexie.py # 后端检索引擎
├── entities/                 # 实体映射数据（运行时生成）
├── references/
│   ├── answer-format.md      # 回答模板与纪律
│   └── source-schema.md      # 检索结果字段说明
└── agents/
    └── openai.yaml           # OpenAI GPT Action 配置模板
```

---

## 版本历史

### v3.1（2026-06-08）— 结构化事件索引

- **新增 14 个历史争议事件的结构化索引**（`events/*.yaml`）
  - 覆盖 2011-2026 年的重要争议事件
  - 每个事件包含：概述、阶段时间线、核心帖子链接、关键词、关键人物、各方立场、关联事件
- **SKILL.md 增强**：
  - 添加完整的角色定位和背景认知章节
  - 详细阐述协会组织架构（理事会→主席团→执委会→各部门）
  - 描述 6 大核心制度体系（训练、技术、拉练、暑期、师徒、论坛）
  - 明确 5 个分析视角（制度、历史、组织、人性、传承）
- **用途**：快速定位历史争议事件的完整脉络，避免重复检索

### v3.0（2026-06-08）— 融合检索 + 专题分析

- **新增版面**：车友宝典（bid=3）、一技之长（bid=7），总计 5 版 13.8 万条
- **三路检索**：FAISS 内容 + FAISS 标题（2x 权重）+ SQLite FTS5 BM25（零内存）
- **句子级语义切分**：替换固定 700 字切片，长帖按句边界分割
- **动态查询路由**：自动判断精确/模糊/混合查询，调整检索策略
- **版面权重系统**：工作区 4.5 → 纯净水 1.0，决定分析优先级
- **专题分析模式**（`--analyze`）：共现图 → Louvain 社区发现 → c-TF-IDF 标签 → 事件映射 → 黑话注释
- **实体映射**：从帖子中自动提取路线/问题/职务/年份，建立反向索引
- **黑话映射表**：35 条车协术语，搜索扩展 + 分析时自动注释
- **事件框架**：5 种事件类型定义，帖子自动归类
- **模型统一**：BGE-small-zh-v1.5（512 维，CPU 高效推理）
- **数据规模**：138,001 条切片，约 3.1 万篇帖子

### v2.0（2026-06-07）— 深度复盘管线

- 增强管线（--pipeline）：多查询扩展 → BM25 混合 → 全帖检索 → 内容过滤
- 三层输出结构
- GitHub Releases 数据分发

### v1.0 — 基础检索

- FAISS 向量检索 + 关键词重排序
- 车协工作区/行者足音/纯净水三版
- 109,527 条切片

---

## License

内部使用。数据版权归北京大学自行车协会及原作者所有，请勿用于商业用途。
