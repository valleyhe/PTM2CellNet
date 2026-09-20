# 2026-09-20 归档说明

本目录保存本次审计确认已被当前状态或当前实现实质性取代的文档副本。归档采用
`cp --preserve=timestamps` 完成，原文件仍保留在原路径，未执行删除、覆盖或移动。

## 归档边界

- `project_analysis_20260917.md`：被本次 `project_analysis_20260920.md` 取代；其内容
  不能反映当前工作区未提交的 2026-09-18 决策与代码变更。
- `real_assets_acceptance.md`：第 4 节仍写明没有跨候选共享 prepare CLI，和当前
  `build_shared_prepare_plans` 及 E2E 的共享 `_prepare/` 实现冲突。

## 可恢复性

归档副本和原文件均保留。`ARCHIVE_MANIFEST.md` 记录原路径、原始 mtime、文件大小、
最近 Git 提交及归档理由；恢复时将副本复制回 manifest 中的原路径，并由主会话在
确认后处理 Git 变更。本次未计算或引入新的哈希文件，亦未修改 Git 历史。
