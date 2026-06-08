# Chexie Knowledge · AGENTS.md

本项目可作为一个 **知识检索工具包** 被 AI Agent（Claude Code、Codex CLI 等）加载使用。本文件说明如何配置。

---

## 核心指令

当 Agent 收到车协相关问题时，执行以下流程：

### Step 1：检索

快速搜索（事实性问题）：
```bash
python scripts/search_chexie.py "<问题>" --top-k 8 --json
```

深度复盘（回溯性问题，如"某次拉练出了什么问题"）：
```bash
python scripts/search_chexie.py --pipeline "25双日白河A组出现了什么问题"
```
输出三层结构：目标拉练全貌（问题+方案+反思）→ 同期对照 → 往年经验，每条带原文链接。

参数说明：

| 参数 | 作用 |
|------|------|
| `--top-k N` | 返回 N 条结果（默认 8，快速搜索用；管线模式默认 20） |
| `--json` | JSON 格式输出，含完整来源字段 |
| `--info` | 查看索引状态（结果数量、模型、路径） |
| `--pipeline` / `-p` | 增强管线模式（深度回溯检索 + 内容过滤） |

### Step 2：基于结果回答

**铁律：先不敢编，再分析。** 回答分四阶段：

1. **事实抽取**：从检索结果中提取结构化事实（source_id + 原文短摘），不做归纳
2. **证据绑定**：每个具体判断（人名、时间、职务、因果）必须绑定 `source_id`
3. **答案生成**：只使用事实表写答案。不合并冲突来源，不补全细节
4. **自检**：逐句检查——无 source_id 的删除，冲突的标注，二手索引的替换为一手

**强制绑定 source_id 的字段**：人名、年份、团/组/路线、职务、因果关系、制度变化。

**来源格式**：
```
判断：……
依据：[source_id] + 【版面】《帖子标题》第X楼，作者，时间，链接
确定程度：高/中/低
```

**事件索引是二手材料**。`events/*.yaml` 只能用于定位帖子，事实细节必须回到原帖 chunk。

**禁止事项**：
- 禁止"根据知识库记载"等模糊来源
- 禁止合并冲突来源
- 禁止补全检索结果中不存在的细节
- 检索结果不足时说"证据不足，无法确认"

---

## 输出 JSON 字段

`search_chexie.py --json` 每次返回：

| 字段 | 说明 |
|------|------|
| `results[].source_id` | 稳定来源标识：`bid{N}_tid{N}_floor{N}` |
| `results[].text` | 帖子正文片段 |
| `results[].score` | 原始向量检索得分 |
| `results[].rerank_score` | 关键词加权后的排序分 |
| `results[].source.board` | 版面名（车协工作区/行者足音/纯净水） |
| `results[].source.title` | 帖子标题 |
| `results[].source.floor` | 楼层号 |
| `results[].source.author` | 作者 |
| `results[].source.time` | 发布时间 |
| `results[].source.url` | 论坛回查链接 |
| `results[].source.source_label` | 识别出的会议/通知/总结标签 |
| `results[].source.source_type` | 类型：meeting / notice / finance / route / thread |
| `results[].source.cues` | 线索：proposal / practice / benefit / risk / outcome |

---

---

## 依赖

```bash
pip install faiss-cpu sentence-transformers numpy
```

模型 `BAAI/bge-small-zh-v1.5` 首次运行时自动下载（~200MB）。
