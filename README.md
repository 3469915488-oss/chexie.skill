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

定义了五种常见活动类型（拉练、执委会、出摊、拉练报名、修车讲座），每个事件有固定结构（路线/年份/季节/组别、会议届次、角色与描述、阶段划分、常见问题类型等）。分析模式自动将帖子归类到事件框架中。拉练类型覆盖 26 条路线名关键词和 10 个岗位角色。

### 分析 Prompt（`prompts/*.yaml`）

三段可配置的 LLM prompt 模板，控制分析输出的风格和深度。支持默认分析、争议深入分析、简洁总结三种模式。

### 结构化争议事件索引（`events/controversies.yaml`）

14 个历史争议事件簇的结构化索引（2001-2024），共收录 74 条帖子。每个事件簇包含：
- **概述**：事件背景与核心争议（`summary`）
- **核心帖子**：74 条帖子的 bid/tid/标题/作者/日期/点击量/回复数/角色定位/URL
- **各方立场**：2-3 个立场方及其核心观点、关键作者（`viewpoints`）
- **后续影响**：事件推动的制度或文化变化（`outcomes`）
- **关联事件**：与其他事件簇的交叉引用（`related_controversies`）

14 个事件簇覆盖：创会精神与治理原罪、老会员代际张力、暑期选拔与落选创伤、纪律安全与下坡争议、理事会权力与版务治理、宣传与组织产出、暑期主题与社考定位、2011性别与平权风波、2013治理危机、赞助商关系与共识争论、互评与追队文化危机、论坛存废与平台迁移、成员疏离与风气之忧、借车制度与边界意识。

这些索引用于快速定位历史争议事件的完整脉络，避免重复检索。

---

## 安装

支持 **Linux / macOS / Windows**。所有脚本通过环境变量定位数据和模型，不硬编码路径。

### 方式一：一键安装（推荐）

`install.py` 自动下载数据、安装依赖、构建 BM25、验证安装：

```bash
# Linux / macOS
python install.py --root ~/chexie-knowledge

# Windows (CMD 或 PowerShell)
python install.py --root C:\chexie-knowledge

# 已有数据，只装依赖
python install.py --skip-data
```

安装完成后按屏幕提示设置环境变量即可。

### 方式二：手动安装

#### 依赖

```bash
pip install -r requirements.txt
```

模型 `BAAI/bge-small-zh-v1.5` 首次运行自动下载（约 200MB）。如需自定义模型缓存目录，设置 `CHEXIE_MODEL_DIR` 环境变量。

#### 下载数据

**Linux / macOS**：
```bash
wget https://github.com/3469915488-oss/chexie.skill/releases/download/v3.0.0/chexie_data.tar.gz
tar -xzf chexie_data.tar.gz -C /opt/chexie-knowledge/
```

**Windows**：
从浏览器下载数据包后用 7-Zip 解压到安装目录，或使用 `install.py`（推荐）。

数据文件：faiss_index.bin（270MB）、title_index.bin（38MB）、chexie_fts.db（343MB）、faiss_meta.jsonl（228MB）、entities/\*.json。

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHEXIE_ROOT` | 数据文件目录 | `/opt/chexie-knowledge`（Linux）或 `%APPDATA%\chexie-knowledge`（Windows） |
| `CHEXIE_MODEL_DIR` | 模型缓存目录 | `/opt/wiki/models`（Linux）或 `~/.cache/huggingface`（macOS/Windows） |

Windows 用户设置示例（CMD 管理员）：
```cmd
setx CHEXIE_ROOT "C:\chexie-knowledge"
setx CHEXIE_MODEL_DIR "C:\chexie-knowledge\models"
```

### 验证安装

```bash
python scripts/search_chexie.py --info
```

### Windows 特别注意

- `wget` 和 `tar` 不是 Windows 自带命令，推荐使用 `install.py`（纯 Python，无需额外工具）
- `faiss-cpu` 有 Windows wheel，不要安装 `faiss-gpu`
- 路径使用反斜杠或正斜杠均可（Python pathlib 自动处理）
- 数据包约 850MB 解压后，确保目标盘有足够空间

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
├── events/                   # 结构化事件索引
│   └── controversies.yaml   # 14个争议事件簇（74条帖子，2001-2024）
├── install.py                 # 跨平台一键安装脚本
├── requirements.txt           # Python 基础依赖
├── requirements-full.txt      # Python 完整依赖（含分析功能）
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
│   ├── detect_events.py      # 事件自动检测（多维度评分）
│   ├── controversy_lookup.py # 争议事件索引守卫（只返回候选 tid）
│   └── server_search_chexie.py # 后端检索引擎
├── tests/
│   ├── smoke_test.py         # 冒烟测试（5 场景）
│   └── test_retrieval_quality.py # 检索质量回归测试（22 用例）
├── entities/                 # 实体映射数据（运行时生成）
├── references/
│   ├── answer-format.md      # 回答模板与纪律
│   └── source-schema.md      # 检索结果字段说明
└── agents/
    └── openai.yaml           # OpenAI GPT Action 配置模板
```

---

## 版本历史

### v3.2（2026-06-08）— 工程一致性 + 生成纪律

**工程一致性修复**（基于外部 code review 反馈）：

- **版本事实统一**：SKILL.md、AGENTS.md、install.py、README.md 全部对齐到 v3.0.0 数据、BGE 模型、五大版块 13.8 万条
- **install.py 重写**：新增 `copy_scripts()` 自动复制检索脚本到安装目录；验证路径修正为 `scripts/search_chexie.py`；REQUIRED_FILES 补充 `chexie_fts.db`
- **--info 友好降级**：faiss/numpy/sentence_transformers 改为延迟导入，`--info` 在缺依赖时也能输出索引状态（文件数、大小、FTS5 记录数、依赖检测结果）
- **默认路由收紧**：`detect_query_type()` 默认从 `"mixed"` 改为 `"exact"`，普通查询走快速搜索而非 pipeline
- **pipeline.py 清理**：删除 inline sys.path hack，顶部统一设置
- **新增 smoke test**：`tests/smoke_test.py` 覆盖 --info / 快速搜索 / FTS5 / pipeline / JSON 五个场景

**生成纪律升级**（防止模型把多个正确证据混合成不准确叙述）：

- **身份重写**：「资深决策顾问」→「证据整理员」，铁律"先不敢编，再分析"
- **四阶段生成流程**：事实抽取 → 证据绑定 → 答案生成 → 自检(verifier)
- **强制 source_id 绑定**：人名、年份、团/组/路线、职务、因果关系、制度变化——出现就必须有 `bid{N}_tid{N}_floor{N}` 标识
- **事件索引降级为二手材料**：`events/*.yaml` 只能用于定位帖子，事实细节必须回到原帖 chunk
- **chunk 输入规范**：去重（tid+floor+author）、限流（同帖最多 5 楼层）、元数据前置
- **自检步骤**：unsupported 删除、conflict 标注、secondary_source 替换为一手
- **prompts/*.yaml 全部重写**：三个模板（默认分析/争议分析/简洁总结）均含四阶段格式
- **search_chexie.py 新增 source_id 字段**：FAISS 和 FTS5 两条搜索路径均输出稳定的 `bid{N}_tid{N}_floor{N}` 标识

### v3.3（2026-06-08）— 争议事件索引重构 + 活动框架优化

- **`events.yaml` 重写**：活动类型事件框架全面优化
  - 拉练类型扩展至 26 条路线名关键词、10 个岗位角色（含描述）、8 个阶段
  - 新增 season/duration/location/quota/deadline 等字段
  - 执委会增加 council/session 正则，出摊增加 location 字段
- **`events/controversies.yaml` 新建**：14 个争议事件簇统一索引
  - 覆盖 2001-2024 年，共收录 74 条帖子（含 bid/tid/作者/日期/clicks/replies/角色定位/URL）
  - 每个事件簇含 summary、viewpoints（2-3 方立场+关键作者）、outcomes、related_controversies 交叉引用
  - 替代旧版 14 个独立 YAML 文件，统一 schema 便于程序化读取
- **SKILL.md 增强**：
  - 添加完整的角色定位和背景认知章节
  - 详细阐述协会组织架构（理事会→主席团→执委会→各部门）
  - 描述 6 大核心制度体系（训练、技术、拉练、暑期、师徒、论坛）
  - 明确 5 个分析视角（制度、历史、组织、人性、传承）
- **代码层守卫**：`scripts/controversy_lookup.py` 强制争议索引只返回候选 tid，防止二手材料泄漏进事实层
  - `validate_no_text_leakage()` 守卫函数，禁止超长文本字段
  - `analyze_topic.py` 添加注释明确 events.yaml vs controversies 的使用边界
- **检索质量回归测试**：`tests/test_retrieval_quality.py`（22 用例）
  - 10 个固定查询集 + 期望命中 tid，覆盖 14 个争议事件簇
  - source_id 格式校验、去重检查、中文 FTS5 分词测试
  - controversy_lookup 守卫测试（导入/返回格式/泄漏检测/tid 反查）
- **`agents/openai.yaml` 重写**：从 8 行占位扩展为完整 OpenAPI 3.0.3 GPT Action spec
  - 2 个端点（`/search`、`/controversy-lookup`）+ API Key 鉴权
  - 完整请求/响应 schema、错误码（400/404/429/500）、速率限制
- **删除旧版 14 个独立事件 YAML 文件**，统一为 `events/controversies.yaml`

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
