# 2026-04-11 数据检测报告归档

本目录保存三个未纳入 Git 的历史运行报告。它们的下载状态、数据规模、脚本路径和质量结论
只对应 2026-04-11 的本地环境，不能作为当前数据 release、训练输入或发布验收依据。

| 文件 | 原始路径 | 归档原因 |
|---|---|---|
| `DOWNLOAD_REPORT.md` | `data/raw/DOWNLOAD_REPORT.md` | 外部源下载状态和数量是一次性快照 |
| `INTEGRATION_REPORT.md` | `data/processed/INTEGRATION_REPORT.md` | 整合版本、命令和规模已由当前 manifest 工作流取代 |
| `VERIFICATION_REPORT.md` | `data/processed/VERIFICATION_REPORT.md` | 质量结论只适用于当时的数据与代码 |

当前数据资产状态以 [`data/manifests/datasets.yaml`](../../../../data/manifests/datasets.yaml)、
[`docs/DATA_UPDATE_WORKFLOW.md`](../../../../docs/DATA_UPDATE_WORKFLOW.md) 和
[`docs/guides/data_integration.md`](../../../../docs/guides/data_integration.md) 为准。完整哈希、mtime
和归档原因见 [`archive/20260808/MANIFEST.md`](../../MANIFEST.md)。
