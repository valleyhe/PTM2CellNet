# PTM2CellNet 项目代码修复报告（2026-08-22）

## 1. 任务范围与结论

本次任务属于：Bug 修复、配置/CI 修复、数据处理脚本修复、测试补充、端到端状态审查。

本报告以当前根目录最新分析报告、已批准的 DAVF × PerturbGen 方案和 `lessons.md` 为依据：

- `[R1]` `project_analysis_20260822.md`：重点使用 §1.2–§1.3、§3.2–§3.4、§4.4、§5.2–§5.3、§7.2。
- `[R2]` `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`：重点使用 §4.5、§5.4、§7.2–§7.3、§9–§10。
- `[L1]` `lessons.md`：重点使用 L-2026-0822-03 至 L-2026-0822-06，以及 L-2026-0821-02。

结论：三轮修复已完成，代码回归通过；TD-N-24、TD-N-10/B1、TD-N-25 和实际 PerturbGen 静态 embedding 导出链已处理。当前用户提供的 checkpoint 已完成“明确 tensor key + 上游 pickle 词表 + 行号对齐 + safetensors/manifest 导出”验证。

但“完整 E2E 训练/推理”仍未完成。Gate-0 仍被独立 Python 3.11 外部环境、合规 normal/disease donor 队列、离线真实 PerturbGen inference/perturb smoke 阻塞；DAVF runtime 尚未接入 PerturbGen 资产，Gate-E、M6 和 M7 不能宣称通过。这与 `[R2]` §7.3 的停止条件和 `[L1]` 的禁止猜测原则一致。

`[R1]` §3.2 中“encoder 权重缺失”的旧记录已被当前用户提供的 `perturbgen_ckpt/` 和本轮实际导出验证部分更新：权重、tensor key、词表行对齐已经有证据；其余 M0 条件仍未满足。

## 2. 三轮修复记录

### 第一轮：修复 GeneMapper 外部网络无限挂起（TD-N-24）

| 项目 | 内容 |
|---|---|
| 问题 | 可选依赖 `UniProtMapper.ProtMapper` 没有 HTTP timeout，内部轮询可能永久等待；它优先于仓库自带的有 timeout mapper，生产 `/predict` 的 `gene_symbol` 路径和 DAVF 集成测试都可能挂死。根因来自 `[R1]` §1.3、§5.2 的实测链路。 |
| 计划 | 只给可选外部依赖增加明确的 90 秒调用上限；超时沿用现有失败语义返回 `None`；不切换到另一套 mapper，避免悄悄改变映射语义。 |
| 修改 | `src/analysis/gene_mapper.py:142-187` 增加受限 daemon worker 和 `threading.Event`；`:211-227` 统一单个/批量调用；`:268-327` 接住 timeout 和 requests 异常。 |
| 回归测试 | `tests/unit/analysis/test_gene_mapper.py` 新增单个映射、批量映射两个“黑洞依赖”测试，断言快速返回、缓存 `None`、且没有 fallback。 |
| 单元/集成结果 | 修复后 GeneMapper 相关单元 + DAVF 挂起回归共 28 passed；其中真实挂起路径在网络隔离下正常退出。最终聚焦命令结果为 `28 passed, 12 warnings, 3.90s`。 |
| 未解决 | Python 线程不能强制终止第三方内部阻塞；当前调用方已被 90 秒边界保护，但如果生产环境持续高并发触发黑洞，仍需后续把外部 mapper 放入可杀进程边界或显式改用本地 mapper。当前不添加静默 fallback。 |

关键实现位置：

```python
if not completed.wait(timeout=_EXTERNAL_MAPPER_TIMEOUT_S):
    raise TimeoutError(...)
```

这是调用边界，不是“等待更久”的补丁；超时后调用方立即进入已有错误处理路径。

### 第二轮：补齐 CI analysis 依赖与离线网络契约（TD-N-10、B1、TD-N-25）

| 项目 | 内容 |
|---|---|
| 问题 | `[R1]` §3.2、§5.2 指出：默认 CI 没有安装 `requirements-analysis.txt`，依赖 `anndata` 的数据契约测试可能只 skip；CI 和本地测试也没有把 Hugging Face 离线语义写成显式契约。 |
| 计划 | 增加独立 `analysis` job，安装分析依赖并执行数据准备、结果解析和 PerturbGen mock 流程；核心 CI 与真实资产 workflow 都显式设置三个离线变量。 |
| 修改 | `.github/workflows/ci.yml` 增加 `analysis` job，Python 3.10 安装 `requirements-dev.txt` 和 `requirements-analysis.txt`；核心测试 job 设置 `HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`HF_DATASETS_OFFLINE`。`.github/workflows/perturbgen-real-assets.yml` 安装分析依赖并补 `HF_DATASETS_OFFLINE`。 |
| 契约测试 | 新增 `tests/unit/test_ci_workflow_contract.py`，静态检查 job、依赖和离线变量不能被删掉。 |
| 验证结果 | YAML 解析通过；CI contract + data prep + results 共 25 passed；mocked PerturbGen integration 3 passed。完整离线测试也通过。 |
| 未解决 | 尚未在 GitHub runner 上实际触发 workflow；远端仍要求有可用的预热 HF cache。真实资产测试按设计在没有真实配置时 skip，不能把本地 mock 结果写成 Gate-4。 |

### 第三轮：让实际 PerturbGen checkpoint 可导出并验证行对齐

| 项目 | 内容 |
|---|---|
| 问题 | 用户提供的真实 checkpoint 使用 `state_dict.transformer.token_embedding.weight` 这一“嵌套 state_dict + flat dotted key”形式，词表是上游 `.pkl`，原 exporter 只接受 JSON 且无法按该显式 key 取矩阵。 |
| 计划 | 只接受用户明确给出的 tensor key；支持受限 builtin-only pickle 词表；保留原有维度、有限值、唯一 token ID 和 manifest 校验；不搜索、不猜 key。 |
| 修改 | `scripts/export_perturbgen_gene_embeddings.py:39-61` 支持嵌套 mapping 内的 flat dotted key；`:64-97` 增加受限 `_VocabularyUnpickler` 和 `.pkl/.pickle` 读取；`:129-155` 在 manifest 和 CLI 中记录词表格式。 |
| 测试 | `tests/unit/scripts/test_export_perturbgen_gene_embeddings.py` 新增上游 pickle 词表和 flat dotted key 回归。专用 exporter 单元 9 passed，PerturbGen mocked integration 5 passed。 |
| 实际结果 | 使用 `perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt` 导出到 `outputs/perturbgen/embedding_asset_20260822/`；得到矩阵 `(18967, 768)`、`torch.float32`、tensor key `state_dict.transformer.token_embedding.weight`、词表 18967 项。对 `ENSG00000187634`、`ENSG00000188976`、`ENSG00000187961` 做 checkpoint 原始行与导出行逐元素相等校验，全部通过。 |
| 未解决 | 这只证明静态 embedding 资产契约，不证明 PerturbGen 模型能在独立环境离线初始化、执行 perturb 或被 DAVF 消费。 |

## 3. 文件级修改清单

| 文件 | 修改目的 | 影响范围 |
|---|---|---|
| `.gitignore` | 放行 `project_repair_*.md`，使本报告可被版本控制发现；不放行模型和 `outputs/`。 | 报告交付，不改变运行逻辑。 |
| `.github/workflows/ci.yml` | 增加分析依赖 job，固化 HF 离线变量。 | PR CI 的数据契约、结果解析和 mock PerturbGen 检查。 |
| `.github/workflows/perturbgen-real-assets.yml` | 安装 `requirements-analysis.txt`，补齐 datasets 离线变量。 | 真实资产门禁的依赖安装和网络行为。 |
| `src/analysis/gene_mapper.py` | 给第三方 mapper 调用加硬超时，保持无 fallback。 | GeneMapper 单个/批量映射及其 DAVF/API 上游调用链。 |
| `scripts/export_perturbgen_gene_embeddings.py` | 接受上游 pickle 词表和明确的 nested/flat tensor key，生成严格资产 manifest。 | 用户提供的 PerturbGen encoder checkpoint 导出。 |
| `tests/unit/analysis/test_gene_mapper.py` | 增加黑洞网络回归。 | 防止外部 mapper 再次无限挂起。 |
| `tests/unit/scripts/test_export_perturbgen_gene_embeddings.py` | 增加真实格式导出回归。 | 防止 pickle 词表或 state_dict key 解析回退。 |
| `tests/unit/test_ci_workflow_contract.py` | 增加 workflow 静态契约测试。 | 防止 CI 依赖和离线变量被误删。 |
| `project_repair_report_20260822.md` | 保存本报告。 | 记录三轮修复、证据、阻塞项和后续计划。 |

## 4. 验证结果

### 4.1 最终代码门禁

聚焦单元命令：

```bash
unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest \
  tests/unit/analysis/test_gene_mapper.py \
  tests/unit/test_gene_mapper.py \
  tests/unit/integration/test_ptm_gene_mapper.py \
  tests/unit/test_ci_workflow_contract.py \
  tests/unit/integration/perturbgen/test_data_prep.py \
  tests/unit/integration/perturbgen/test_results.py \
  tests/unit/scripts/test_export_perturbgen_gene_embeddings.py \
  tests/unit/integration/perturbgen/test_embedding_export.py \
  tests/unit/models/test_perturbgen_embedding.py \
  -q -p no:cacheprovider
```

聚焦集成命令：

```bash
unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest \
  tests/integration/test_davf_pipeline.py::TestDAVFPipelineIntegration::test_multiple_ptm_types_batch \
  tests/integration/test_perturbgen_pipeline_mocked.py \
  tests/integration/test_perturbgen_dual_path_mocked.py \
  -q -p no:cacheprovider
```

| 检查 | 实际命令/结果 |
|---|---|
| 聚焦单元 | `unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest ... -q -p no:cacheprovider`：74 passed，15 warnings，7.51s。覆盖 GeneMapper、CI contract、data prep/results、exporter、embedding loader。 |
| 聚焦集成 | 同样的离线环境执行 DAVF 多 PTM、PerturbGen mocked pipeline、dual path：6 passed，12 warnings，3.97s。 |
| 全量测试 | `unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest tests -q -p no:cacheprovider`：收集 2331 项，2315 passed、16 skipped、47 warnings，520.88s。真实资产相关 skip 是无真实配置时的预期行为。 |
| 编译 | `python -m compileall -q src scripts tests`：退出码 0。 |
| Ruff | `ruff check src scripts tests`：All checks passed。 |
| 类型 | `python -m mypy src`：23 errors in 8 files，与 `[R1]` 最新基线一致；本轮 GeneMapper 改动没有增加错误。剩余错误集中于既有 PerturbGen 类型标注、runner 和 API 路由。 |
| 环境依赖 | `python -m pip check` 失败：缺 `datasketch`、项目要求 `numpy<2` 但当前为 2.4.3、`scgpt` 要求 `scvi-tools<1.0` 但当前为 1.4.3、`ssh-unit` 要求 `torchaudio>=2.5` 但当前为 2.4.1+cu118。该环境问题未由本轮代码引入，不能据此宣称干净发布环境。 |

### 4.2 M0/Gate-0 和完整 E2E 要求复核

| 要求 | 当前证据 | 判定 |
|---|---|---|
| 真实 encoder checkpoint、明确 tensor key | 用户提供 checkpoint；实际导出并校验 `(18967, 768)`，key 为 `state_dict.transformer.token_embedding.weight`。 | 已完成该子项 |
| token dictionary 到 embedding row 对齐 | 上游 pickle 词表读取成功；三个 ENSG 样本做原始行/导出行逐元素校验。 | 已完成静态子项 |
| 独立 Python 3.11 PerturbGen 环境 | 当前环境为 Python 3.12.13，`python3.11` 不存在。 | 阻塞 |
| 离线 PerturbGen 初始化/inference | 当前只有 exporter 和 mock 测试；没有独立环境中的真实初始化证据。 | 未完成 |
| 真实 perturb smoke 与资源基线 | 没有真实 perturb config、双路径 h5ad 产物、P50/P95/RSS/显存 evidence。 | 未完成 |
| 合规 normal/disease donor 队列 | 既有审计记录：30 个本地 h5ad 中 26 可读、4 截断；唯一显式 patient 队列只有 2 人且全为 healthy，不是 normal/disease 配对；合规候选 0。该事实与 `[L1]` 一致。 | 阻塞 |
| DAVF 消费 PerturbGen 资产 | `src/models/davf_inference.py:278` 仍是 `LatentDAVF(latent_config)`；`configs/davf_integration.yaml` 仍有 `geneformer_path`；当前 runtime 没有 PerturbGen asset 注入。 | 未完成，阻塞 M4 |
| DAVF 重训和 checkpoint schema v2 | 尚未开始，不能用静态导出替代重训。 | 未完成 |
| Gate-E 固定 ≥200 PTM→gene 基准集 | 无固定基准集、无新旧 embedding 配对指标和 bootstrap CI。 | 未完成 |
| Gate-4 真实资产发布证据 | workflow 门禁脚手架存在，但真实证据不存在；本地真实资产测试 skip。 | 未通过 |
| API/测试网络无限挂起 | TD-N-24 回归已通过，28 个相关测试和全量测试均退出。 | 已修复 |

因此当前状态是：

> 工程代码和 mock 端到端链路稳定；真实 PerturbGen → DAVF → 科学验收链路仍未闭环。

## 5. 严重/阻塞问题与后续策略

以下排期沿用 `[R2]` §7.1 的日期；日期是执行窗口，不是对缺失外部资产的承诺。

| 时间窗口 | 问题 | 具体策略 | 完成条件 |
|---|---|---|---|
| 2026-08-23 | 独立运行环境缺失 | 建立 Python 3.11 的独立 PerturbGen 环境或固定容器；锁定源码 commit、CUDA/GPU、依赖版本。不要修改当前混杂环境，不要用 Python 3.12 代替。 | `python --version`、源码 commit、GPU/CUDA、依赖清单均写入 M0 evidence；HF/evaluate 组件可离线初始化。 |
| 2026-08-24～08-25（M0） | Gate-0 仍未满足 | 在上述环境中使用当前已验证的 tensor key 和词表，执行离线初始化、token round-trip、真实 perturb config、最小双路径 workload，并采集耗时、峰值 RSS、显存和产物 manifest。 | M0 六项 evidence 全部存在：权重、独立环境、离线 inference、token/row、真实 perturb、资源基线；任一缺失就保持 BLOCKED。 |
| 2026-08-26～08-30（M1） | 没有合规科学队列 | 获取 raw counts、ENSG、cell type/state、明确 donor/patient 的 normal/disease AnnData；至少 3 个可评估 donor。不能把 sample、batch、replicate 或 CRISPR control 自动解释成 donor。 | cohort manifest 明确记录 donor、状态、样本来源和排除项；preflight 无截断文件。 |
| 2026-08-31～09-03（M2/T3） | 真实 runner 尚未执行 | 用 M0 确认的真实输出路径执行 tokenise、embedding/perturb、结果解析和 resume；把具名输入/输出/hash/资源指标写入单次自洽 evidence。 | 真实环境 smoke 通过，产物路径和 manifest 可重放；不能用多个残缺运行拼成一份 evidence。 |
| 2026-09-10～09-16（M4） | DAVF 仍使用旧 Geneformer 语义 | 按 `[R2]` §4.5 修改 `architectures.py`、`davf_inference.py`、`latent_davf.py`、`ptm_direction_mapper.py`、`davf.py`、配置和 `finetune_davf.py`；引入共享 `GeneVocabularyResolver`/manifest 注入，升级配置 schema v2 和 checkpoint schema；删除 null→随机 embedding 语义。 | runtime 实际读取当前 manifest；缺资产直接报错；DAVF 重训/恢复成功；新增真实资产 contract test。Gate-0 未通过前不动手接入。 |
| 2026-09-17～09-21（M5） | Gate-4 没有真实 evidence | 在 self-hosted GPU workflow 执行双路径真实 smoke，确保两个独立 h5ad、峰值显存、P50/P95、RSS、产物体积都存在。 | mock 和真实门禁同时绿；没有 evidence、过期 evidence 或 skip 就失败。 |
| 2026-09-22～10-02（M6） | 没有科学验收 | 冻结 ≥200 个可追溯 PTM→gene 样本，固定划分，至少 3 seeds，运行 mask/pad/delete/overexpress、≥99 matched null、BH-FDR、bootstrap CI 和 donor-held-out 检验。 | Gate-5/Gate-E 指标满足 `[R2]` §5.4；若新底座劣化，保留旧发布版本并停止迁移。 |
| 2026-10-05～10-06（M7） | Geneformer 仍在主链 | 仅在 Gate-E 和 Gate-5 都通过后移除 Geneformer 主链、旧配置和专属测试，更新迁移指南。 | `rg geneformer` 只剩历史说明或明确兼容边界；发布 evidence 完整。 |

并行的低风险收尾项：

- 在 M4 开始前清理 `[R1]` §5.2 中剩余 23 个 mypy 错误，优先 `src/integration/perturbgen/`、`src/models/gene_vocabulary.py` 和 runner；每次只提交真实类型修复，不用 `ignore-errors` 扩大豁免。
- 为当前环境建立干净的可复现环境：补 `datasketch`，按项目约束使用 `numpy<2`，将 scGPT 与 scvi-tools 版本约束拆到适配的外部环境，并匹配 torch/torchaudio；不在当前共享环境中盲目降级依赖。
- 监控 90 秒 daemon worker 的高频超时资源占用。若真实部署存在高频黑洞，再将第三方 mapper 放到进程边界；不增加会改变映射结果的静默 fallback。

## 6. 可复现命令

```bash
# 全量离线测试
unshare -rn env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  python -m pytest tests -q -p no:cacheprovider

# 静态门禁
python -m compileall -q src scripts tests
ruff check src scripts tests
python -m mypy src

# 实际 checkpoint 导出
python scripts/export_perturbgen_gene_embeddings.py \
  --checkpoint perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt \
  --tensor-key state_dict.transformer.token_embedding.weight \
  --vocabulary ref/Perturbgen-src/perturbgen/pp/token_dict_gftokens_gc95M.pkl \
  --output-dir outputs/perturbgen/embedding_asset_20260822
```

最后一个命令要求输出目录不存在；导出结果应包含 `gene_embeddings.safetensors`、`vocabulary.json` 和 `manifest.json`。本轮已实际执行并完成行对齐验证。

## 7. 最终判定

| 判定项 | 状态 |
|---|---|
| 三轮修复 | 已完成 |
| 单元测试 | 已完成，最终聚焦 74 passed；全量测试 2315 passed |
| 集成测试 | 已完成，最终聚焦 6 passed |
| 真实 checkpoint 静态导出 | 已完成 |
| TD-N-24 | 已修复 |
| TD-N-10/B1、TD-N-25 | 代码和本地契约已修复，远端 workflow 尚未实际触发 |
| 完整 PerturbGen 真实 inference/perturb | 未完成 |
| DAVF 真实底座迁移、重训、Gate-E | 未完成，受 Gate-0 阻塞 |
| 报告 | 已生成：`project_repair_report_20260822.md` |

本轮不把 mock 通过、静态 embedding 导出或全量单元测试通过写成科学 E2E 完成；这是当前最重要的边界。
