# 车协知识库 · Chexie Knowledge

> **北大车协往年经验知识库**——检索 BBS 工作区、行者足音、纯净水三大版块的 10 万+ 帖子，让 AI 基于有来源、有楼层、可回查的论坛记录回答你的问题。

---

## 它能做什么

问一个问题，得到一份带来源的答案。比如：

**问：** 往年远征报名费多少？报名流程是什么？

**答：** 2025 年远征报名费 20 元/人，报名方式为填写报名问卷 + 微信转账给温瑶，备注姓名和 ID，同时需在论坛提交自述。报名时间截止到 5 月 22 日 24:00。落选暑期的会员不退还报名费。理事会 2025 年 4 月 18 日发布通知。

——来源：【车协工作区】《理事会关于2025暑期远征报名的通知》第29楼，理事会，2025-04-18，链接

覆盖范围：
- **组织管理**：执委会、理事会、拉练总结、职务安排、财务预算
- **骑行活动**：远征、双日、单日、前站
- **技能培训**：春训体测、检车标准、队医贴士
- **装备团购**：价目、规格、组织流程
- **经典路线**：十渡、白羊沟、妙峰、禅房、凤凰岭、官厅、白河……

---

## 工作原理

```
用户提问 → 向量检索 (FAISS) → 关键词重排序 → 召回 Top-K 结果 → AI 依据来源回答
         ↕ 深度复盘（--pipeline）：多查询扩展 → BM25 混合检索 → 全量帖子拉取 → 内容过滤 → 三层输出
```

- **嵌入模型**：`shibing624/text2vec-base-chinese`（中文语义匹配）
- **向量索引**：FAISS（余弦相似度，109527 条 evidence）
- **重排序**：关键词 Boost + 版面加权 + 事件精确匹配
- **执行方式**：一条 Python 脚本，自动检测本地/远程执行
- **回答纪律**：每个结论必须附版面、标题、楼层、作者、时间、链接

---

## v2.0 更新说明（2026-06-07）

本次更新解决了向量检索的一个核心缺陷：长篇帖子（队长总结、执委会记录）被切成多个 chunk 存入 FAISS 后，**中间的（时间线、情况说明、问题与思考）经常丢失**。库里明明有数据，但检索不出来。

### 新增

| 功能 | 说明 |
|------|------|
| **增强管线** (`--pipeline`) | 多 query 扩展 → FAISS + BM25 混合检索 → RRF 融合 → 全量帖子拉取 → 内容过滤 |
| **三层输出** | 目标拉练全貌（问题+方案+反思）→ 同期对照 → 往年经验，每条带原文链接 |
| **内容过滤** | 自动提取问题/解决方案/反思，过滤致谢和个人废话 |
| **BM25 关键词索引** | 构建脚本 `build_bm25.py`，`--bm25` 开关启用 |
| **开箱即用** | 数据文件独立发布，clone + 解压即可用 |

### 安装变更

数据文件（faiss_index.bin + faiss_meta.jsonl + bm25_index.jsonl）从 Google Drive 迁移到 **GitHub Releases**，与代码仓库分离发布，避免仓库臃肿。

---

## 快速使用

### 快速检索

```bash
python scripts/search_chexie.py "远征报名流程" --top-k 8 --json
```

### 深度复盘检索

适用于"某次拉练出了什么问题""怎么解决的""有什么经验教训"这类回溯性问题。
管线自动识别目标帖子、提取问题与解决方案、匹配同期和历年对照。

```bash
python scripts/search_chexie.py --pipeline "25双日白河A组出现了什么问题"
```

输出三层：目标拉练全貌（问题+方案+反思）→ 同期对照 → 往年经验。每条带原文链接。

### 全部参数

| 参数 | 说明 |
|------|------|
| `query` | 检索问题 |
| `--top-k N` | 返回结果数（默认 8，快速搜索；管线模式默认 20） |
| `--candidate-k` | 候选池大小（默认 top-k × 30） |
| `--json` | JSON 格式输出，含完整来源字段 |
| `--info` | 查看索引状态 |
| `--pipeline` / `-p` | 增强管线模式（深度回溯检索 + 内容过滤） |
| `--bm25` | 启用 BM25 关键词搜索（需先运行 build_bm25.py） |
| `--verbose` / `-v` | 详细输出（管线诊断信息） |

### Top-K 调优建议

| 问题类型 | top-k |
|----------|-------|
| 简单事实查询（"报名费多少"） | 8 |
| 具体路线/事件/争议 | 12-16 |
| 回溯复盘/多方案比较 | 16-20 |
| 确认性判断（结果疑似偏差时加 k 重试） | 12+ |

### 作为 AI Agent 使用

**Hermes Agent**

```bash
# skill 已内置，直接对 Hermes 提问即可
hermes> 十渡拉练报名人数太多怎么处理
```

**Claude Code / Codex CLI**

项目根目录放 `AGENTS.md`，自动被识别。详见该文件。

---

## 安装

### 方式一：Git Clone + 数据下载（推荐）

```bash
git clone https://github.com/3469915488-oss/chexie.skill.git
cd chexie.skill

# 下载数据文件（~450MB）
wget https://github.com/3469915488-oss/chexie.skill/releases/download/v2.0.0-pipeline/chexie_data.tar.gz
tar -xzf chexie_data.tar.gz

# 安装依赖
pip install -r requirements.txt

# 可选：构建 BM25 关键词索引（约 2.5 分钟，仅首次需要）
python scripts/build_bm25.py
```

数据文件包含：faiss_index.bin（向量索引）、faiss_meta.jsonl（帖子元数据）、bm25_index.jsonl（关键词索引缓存）。

### 方式二：通过 Hermes 自动安装

```bash
hermes config set skill.chexie-knowledge.enabled true
```

### 方式三：从 Releases 下载完整包

从 [Releases 页面](https://github.com/3469915488-oss/chexie.skill/releases) 下载 `chexie_data.tar.gz` 和数据文件，解压到同一目录。

### 依赖

Python ≥ 3.9：

```
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
numpy>=1.21.0
jieba>=0.42.1
rank-bm25>=0.2.2
```

模型 `shibing624/text2vec-base-chinese` 首次运行自动下载（~500MB）。国内使用 HuggingFace 镜像 `hf-mirror.com`。

---

## 目录结构

```
chexie.skill/
├── SKILL.md                  # Hermes Agent Skill 定义
├── README.md                 # 本文件
├── AGENTS.md                 # Claude Code / Codex CLI 配置
├── faiss_index.bin           # FAISS 向量索引（109527 条）
├── faiss_meta.jsonl          # 元数据（帖子标题/作者/链接）
├── bm25_index.jsonl          # BM25 关键词索引缓存（可选）
├── requirements.txt          # Python 依赖
├── build_bm25.py             # BM25 索引构建脚本
├── scripts/
│   ├── search_chexie.py      # 检索入口（快速搜索 + 深度复盘管线）
│   ├── server_search_chexie.py  # 后端检索引擎
│   ├── pipeline.py           # 增强管线（内容过滤 + 三层输出）
│   └── build_bm25.py         # BM25 索引构建
├── references/
│   ├── answer-format.md      # 回答模板与纪律
│   └── source-schema.md      # 检索结果字段说明
├── agents/
│   └── openai.yaml           # OpenAI GPT Action 配置模板
└── ...
```

---

## 数据来源

- **车协工作区**（bid=1）：理事会通知、执委会记录、拉练通知/总结、财务公示、职务培训
- **行者足音**（bid=2）：个人骑行游记、自述、路线体验、装备评价
- **纯净水**（bid=4）：约车帖、非正式讨论、散装经验

总计 **109527 条**检索切片，覆盖时间跨度 2006-2026。

---

## 回答纪律

1. **有来源才说话**——没有检索到材料的问题，明确说"知识库中未找到足够依据"。
2. **来源格式不可省略**——每条关键判断必须附【版面】《标题》第X楼，作者，时间，链接。
3. **区分正式材料和个人经验**——通知/执委会记录归为正式制度，行者足音/纯净水标明为个人讨论。
4. **多个来源冲突时分别列出**——不强行合并成一个确定结论。
5. **覆盖全维度**——结论、往年做法、替代方案、利弊、可参考建议、来源，六点齐全。

---

## License

内部使用。数据版权归北京大学自行车协会及原作者所有，请勿用于商业用途。
