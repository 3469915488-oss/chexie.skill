# 检索结果字段

`scripts/search_chexie.py --json` 返回：

- `question`：原始问题。
- `results`：检索命中的证据列表。
- `text`：可引用正文片段。
- `score`：原始向量相关度。
- `rerank_score`：结合关键词后的排序分。
- `source.board`：版面名。
- `source.tid`：帖子 ID。
- `source.title`：帖子标题。
- `source.floor`：楼层。
- `source.author`：楼层作者。
- `source.time`：楼层发布时间。
- `source.url`：可回查链接。
- `source.source_label`：标题或正文中识别出的会议、通知或总结标签。
- `source.source_type`：如 `meeting`、`notice`、`finance`、`route`、`thread`。
- `source.cues`：正文线索，例如 `proposal`、`practice`、`benefit`、`risk`、`outcome`。
