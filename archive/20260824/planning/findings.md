# 代码修复研究发现

## 目标与口径
- 当前日期：2026-08-16。
- 目标文档：`project_analysis_20260816.md`。
- 最终交付：根目录 `project_repair_report_20260816.md`。

## 分析文档证据
- `project_analysis_20260816.md` §3.3、§7.4、§8（尤其 §8 表格行 311–322）确认：标准链路已工程闭环，但 P0 受控数据/I-06 产品决策仍阻塞；本轮优先选择可由代码独立推进的 N04 ensemble、GSE90546 下游 manifest、N07 multitask 热路径。
- §8 策略要求：N04 先行为快照再公共实现；GSE90546 登记 per-GSM NPZ/聚合 TSV 并补组装契约；N07/N08 先 benchmark 再向量化并保持指标/限额不变。
- §7.2/§7.3 判定基准：T-01–T-09、I-01–I-05 为标准工程链路；T-10 与 I-06 仍不可在缺少外部数据/产品决策时宣称完成。

## 当前代码证据
- 基线环境：Python 3.12.13，pytest 9.0.3；CodeGraph 已索引 354 个文件、7470 个节点、8292 条边。
- `scripts/ensemble_predict.py` 的 `ModelEnsemble` 与 `src/models/ensemble.py` 的 `PTM2CellNetEnsemble` 聚合逻辑分叉；现有快照 10 例通过，但公共实现未复用。
- `scripts/import_norman_adamson.py` 已生成 GSE90546 per-experiment NPZ/TSV，但 `import_manifest.json` 只嵌入原始解析报告，缺少对产物 SHA-256 的显式 artifact catalog；`datasets.yaml` 预处理 outputs 也未声明 GSE90546 产物。
- `src/data/multitask_dataset.py` 的 `MultiTaskPTMDataset.__init__` 和 `MultiTaskPTMDataModule.setup` 各自使用 `DataFrame.iterrows()`，同一行转换逻辑重复。

## 基线测试
- ensemble 单元 + cross-scale 集成：11 passed，8 warnings。
- GSE/manifest 单元 + 集成：32 passed。
- multitask 单元 + cross-scale 集成：10 passed，8 warnings。
- 初始错误：基线命令引用不存在的 `tests/unit/models/test_ensemble.py`，pytest 收集 0 项并返回 4；改用实际存在的脚本快照和集成测试后通过。该错误不影响代码结论。

## 三轮迭代记录
### 迭代一
（待记录。）

### 迭代二
（待记录。）

### 迭代三
（待记录。）

## 系统复核
（待记录。）

## 2026-08-22 DAVF × PerturbGen 重新分析

- v2.0 对现状判断正确：现有 `PTMDirectionMapper.direction` 是 action code，不是 observed expression direction；新契约已强制拆分。
- M0 真实资产不存在，因此最优决策是完成 M1–M3 与严格资产协议，同时停止 DAVF runtime 迁移；任何随机 embedding、猜 checkpoint key 或 silent fallback 都会制造伪完成。
- PerturbGen 上游 CLI 保留下划线参数；tokenized `.dataset` 是目录；checkpoint/h5ad 名含运行时动态部分。现已从锁定 commit 源码冻结“tokenise 路径公式 + 隔离目录唯一 glob + manifest artifact/hash”契约；T3 仅剩真实环境执行证据，不再依赖人工输入动态路径。
- 正式判定新增未扰动模型质量门；缺质量证据、null<99、donor<3、一路不可评估均为 `INCONCLUSIVE`，不是 `FAIL`。
- PerturbGen 真实 h5ad 的 `var` 只有 index，上游不生成 `ensembl_id` 列；契约已保留真实 layers/obs/obsm/varm 结构门并移除错误的 `required_var` 强制项。
- M3 不能只停在库函数：正式重放入口必须显式输入候选 p 值、跨候选做 BH，并把 h5ad、stage manifest、实际 SHA-256 绑定；相对路径统一按输入 JSON 定位，禁止依赖当前工作目录。
- M5 的“有 benchmark JSON”不等于发布证据：正式门要求每个样本本身都含双路径、两个 h5ad 和有效资源采样，不能把多次残缺运行的并集拼成通过。
- Gate-0 复核发现本地 `data/raw/scperturb/` 的 30 个 h5ad 不能替代方案要求的冻结队列：4 个文件截断；唯一显式 `patient` 队列只有 2 人且 disease 全为 healthy。`sample`、`batch`、`replicate` 不得被推断成 donor。
