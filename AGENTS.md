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

回答必须遵循：

1. **有来源才说话。** 检索结果不足时明确说"知识库中未找到足够依据"。
2. **每条关键判断附来源。** 格式：
   ```
   【版面】《帖子标题》第X楼，作者，时间，链接
   ```
3. **区分正式材料和个人经验。** 车协工作区的通知/执委会记录是正式制度，行者足音/纯净水的帖子标明为个人讨论/游记。
4. **来源冲突时分别列出**，不强行合并。

---

## 输出 JSON 字段

`search_chexie.py --json` 每次返回：

| 字段 | 说明 |
|------|------|
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

## 自动检测本地/远程

脚本自动判断执行环境：

- 在云服务器（49.232.25.231）上直接调用本地检索引擎
- 其他环境通过 SSH 调用远程服务器（需 SSH key 配置）

---

## 依赖

```bash
pip install faiss-cpu sentence-transformers numpy
```

模型 `shibing624/text2vec-base-chinese` 首次运行时自动下载（~500MB）。
