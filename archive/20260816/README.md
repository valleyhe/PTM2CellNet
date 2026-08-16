# 2026-08-16 归档批次

本目录保存 2026-08-16 项目代码与文档综合审计中判定为过时/被取代的文档与报告。

## 来源与版本

- 审计基线：`origin/main` = `c76fff881f268f9bd0b39d68db8ba547115ec4cf`（fetch 确认本地一致）
- 本轮代码提交：`32ded55`（Norman/Adamson 导入管线 + OmniPath kinase_substrate 登记，
  直接提交到 `main`，无合并冲突）
- 归档日期：2026-08-16（Asia/Shanghai）
- 详细文件判定与证据：[`MANIFEST.md`](MANIFEST.md)
- 当前权威综合报告：仓库根目录 `project_analysis_20260816.md`

## 归档决策

`32ded55` 将 norman_adamson、kinase_substrate、string、bioplex、regnetwork 五个
数据集从 `controlled` 翻转为 `implemented`，测试基线变为 1716 passed / 4 skipped，
使 2026-08-08/09 报告中"8 个受控输入无路径/数据所有者职责"的中心结论与
1897/1918/1920 系列测试计数失去现时性：

1. 根目录两份 2026-08-09 报告被新的 `project_analysis_20260816.md` 取代，移入 `reports/`；
2. `docs/` 直下三份 8-8 基线报告移入 `docs/`，原路径留重定向短页以保持链接兼容；
3. 其余活动文档、入口短页与指南经逐项核实保留原路径（台账见 MANIFEST 第 4 节）。

## 晚间批次追加（同日 21:00）

晚间综合处理会话（5 组代码提交 `bcb9cc9`→`aa8b15b` 后）追加归档 3 个文件：

4. `detection_artifacts/coverage.xml`、`coverage.json`：8-8/8-9 生成的覆盖率快照，
   早于本轮代码与测试基线（1919 passed / 5 skipped），被根目录 2026-08-16 13:50
   的 `.coverage` 取代；
5. `reports/project_analysis_20260816_v3_repair_iterations.md`：修复迭代任务的
   中间版报告（v3.0，从未提交），权威内容由根目录 `project_repair_report_20260816.md`
   承载，综合分析结构由同日 v4.0 恢复，本中间版归档留痕。

历史文件的原始版本和修改记录可通过 `git log --follow -- <path>` 追溯；本批次不
删除任何历史内容。
