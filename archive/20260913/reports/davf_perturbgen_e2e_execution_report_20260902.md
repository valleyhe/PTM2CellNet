# DAVF × PerturbGen E2E 代码修改与测试报告

**日期**：2026-09-02  
**项目**：PTM2CellNet

## 1. 任务判断

本任务属于新功能实现、代码重构、数据处理、训练入口、集成测试、配置同步和需求转代码设计。

分析基于以下项目输入：

- 当前 DAVF、scVI、PerturbGen runner 和方向 gate 代码；
- 本机真实 KO/KD scVI 与 DAVF checkpoint；
- 当前 PerturbGen embedding asset；
- 现有训练数据、配置、测试和项目 `lessons.md`；
- CodeGraph 结构分析和只读代码审查结果。

## 2. 第一性原理分析

项目的最终目标不是让 DAVF 和 PerturbGen 预测同一个结果，而是拆分为两个严格不同的问题：

```text
PTM site prediction
        ↓
DAVF：预测目标基因表达变化方向
        ↓
方向 gate：PTM proposal + DAVF direction + independent expression
        ↓
PerturbGen：预测全基因表达效用、rescue、signature 和 pathway 结果
        ↓
source_intervention AND within_state
```

三个坐标系必须保持独立：

| 对象 | 作用 | 索引来源 |
|---|---|---|
| DAVF gene token | DAVF 条件输入 | 已验证的 PerturbGen vocabulary |
| scVI latent | DAVF 的状态空间 | 当前 route 对应的 scVI encoder |
| scVI decoder gene index | 读取目标基因表达变化 | `adapter.gene_names` |

KO 和 KD 不能共用未经条件化的 latent 坐标。两条路径应分别加载自己的 scVI 模型和 DAVF checkpoint，最后只能按 canonical Ensembl ID 合并。

## 3. 已完成的代码修改

### 3.1 DAVF 数据与训练链路

新增或完善了以下入口：

```text
prepare_davf_scperturb.py
    → train_davf_scvi.py
    → build_davf_scperturb_pairs.py
    → train_latent_davf.py
    → validate_latent_davf_checkpoint.py
```

主要约束：

- 固定 `latent_dim=64`；
- 固定 `num_genes=4018`；
- 从真实 scPerturb H5AD 构造 latent pair；
- control latent 按 batch 计算，且 train/val/test control cell 不重叠；
- 目标基因按 target label 划分，避免同一目标泄漏到不同 split；
- `gene_ids` 只能是 PerturbGen token index；
- scVI gene order 必须与当前 adapter 完全一致；
- 缺失 token、缺失方向、轴不一致和非有限 latent 直接失败。

### 3.2 当前 DAVF checkpoint contract

正式方向证据只接受：

- `schema_version=2`；
- `model_type=LatentDAVF`；
- `latent_dim=64`；
- `num_genes=4018`；
- 完整且严格匹配的 `LatentDAVF` state dict；
- 当前 PerturbGen embedding asset 的完整 manifest；
- 当前 scVI 的有序 gene vocabulary；
- 冻结的 PerturbGen gene embedding table。

旧的 `delta_mlp` 或 legacy DAVF checkpoint 只能用于历史 feature path，不能生成正式 direction evidence。

### 3.3 严格的 DAVF → PerturbGen 编排器

新增 [orchestrator.py](../../src/integration/perturbgen/orchestrator.py)，实现：

1. 读取显式 KO/KD route；
2. 使用 verified PerturbGen vocabulary 构造 DAVF token 输入；
3. 调用 `DAVFInferenceModule.predict_expression_direction()`；
4. 从 `adapter.gene_names` 解析 scVI decoder 目标列；
5. 执行 PTM proposal、DAVF 方向和独立表达方向三方 gate；
6. 只有 gate 通过才创建 `PerturbGenInvocation`；
7. 为每个候选生成 `source_intervention` 与 `within_state` 两条独立 PerturbGen stage plan；
8. 由现有 runner 管理 GPU lock、artifact、manifest 和 resume。

未知目标、方向冲突、route 不匹配和 symbol/Ensembl 冲突都会直接失败或返回 `inconclusive`，不会使用零向量或随机结果补救。

### 3.4 E2E CLI 与报告合并

新增：

- [run_davf_perturbgen_e2e.py](../../scripts/run_davf_perturbgen_e2e.py)
- [merge_davf_perturbgen_reports.py](../../scripts/merge_davf_perturbgen_reports.py)

E2E CLI 默认只执行真实 DAVF/scVI 方向推理和 gate。只有显式传入 `--run-perturbgen` 才运行外部六阶段 PerturbGen。

KO/KD 报告最终按 canonical Ensembl ID 合并，合并入口强制要求一份 KO 报告和一份 KD 报告。

## 4. 当前配置与真实资产

正式配置：

| 路径 | 配置 |
|---|---|
| KO | `configs/davf_ko.yaml` |
| KD | `configs/davf_kd.yaml` |
| 历史 DAVF feature path | `configs/davf_integration.yaml` |
| 当前通用配置 | `configs/davf_current.yaml` |

`configs/davf_integration.yaml` 保持历史默认值，不表示旧 checkpoint 已经与当前 scVI 坐标匹配。

当前真实资产：

| 路径 | scVI | DAVF |
|---|---|---|
| KO | `checkpoints/scvi/davf_ko_dixit` | `checkpoints/davf/davf_ko_dixit/best_model.pt` |
| KD | `checkpoints/scvi/davf_kd_nadig` | `checkpoints/davf/davf_kd_nadig/best_model.pt` |
| 共用 embedding asset | `outputs/perturbgen/embedding_asset_20260822` | 18967×768 |

两个 scVI 模型均为 4018 genes × 64 latent，两个 DAVF checkpoint 均为当前 schema v2 的 `LatentDAVF`。

## 5. 可执行运行顺序

### 5.1 验证 checkpoint、scVI 和 embedding

```bash
python scripts/validate_latent_davf_checkpoint.py \
  --checkpoint checkpoints/davf/davf_ko_dixit/best_model.pt \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822

python scripts/validate_latent_davf_checkpoint.py \
  --checkpoint checkpoints/davf/davf_kd_nadig/best_model.pt \
  --scvi-model checkpoints/scvi/davf_kd_nadig \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822
```

### 5.2 执行真实 DAVF 方向 gate

候选 JSON 必须为每个候选指定真实 context cell：

```json
{
  "context_h5ad": "data/processed/context.h5ad",
  "candidates": [
    {
      "context_cell_index": 0,
      "gene_symbol": "STAT3",
      "ensembl_id": "ENSG00000168610",
      "position": 12,
      "ptm_type": "phosphorylation",
      "proposed_direction": "down",
      "site_probability": 0.95,
      "provenance": "ptm-site-model/run-1",
      "cell_type": "K562",
      "ptm_context": "STAT3:S12",
      "observed_log2fc": -1.0,
      "observed_fdr": 0.01,
      "observed_direction": "down"
    }
  ]
}
```

执行 KO 方向推理：

```bash
python scripts/run_davf_perturbgen_e2e.py \
  --davf-config configs/davf_ko.yaml \
  --candidate-spec path/to/ko_candidates.json \
  --output outputs/davf_perturbgen/ko_report.json
```

### 5.3 执行 PerturbGen 六阶段

```bash
python scripts/run_davf_perturbgen_e2e.py \
  --davf-config configs/davf_ko.yaml \
  --candidate-spec path/to/ko_candidates.json \
  --perturbgen-config configs/integration/perturbgen.yaml \
  --perturbgen-output-root outputs/perturbgen/e2e \
  --run-perturbgen \
  --output outputs/davf_perturbgen/ko_report.json
```

两个 perturb 阶段分别使用：

- `source_intervention`：从 reference/control 状态施加干预；
- `within_state`：在目标状态内部评估稳定性。

### 5.4 合并 KO/KD 结果

```bash
python scripts/merge_davf_perturbgen_reports.py \
  --reports outputs/davf_perturbgen/ko_report.json \
            outputs/davf_perturbgen/kd_report.json \
  --output outputs/davf_perturbgen/ko_kd_merged.json
```

## 6. 已执行验证

| 验证项 | 结果 |
|---|---|
| DAVF/PerturbGen 聚焦回归 | 157 passed / 2 skipped |
| 真实 KO/KD CUDA bridge | 3 passed |
| KO checkpoint 验证 CLI | `ok=true` |
| KD checkpoint 验证 CLI | `ok=true` |
| embedding asset | 18967×768，匹配 |
| scVI 模型 | 4018×64，匹配 |
| `ruff check src scripts tests` | 通过 |
| `python -m compileall -q src scripts` | 通过 |
| 新增 CLI help | 全部返回码 0 |
| `git diff --check` | 通过 |

真实 CUDA bridge 覆盖了：

- KO / CREB1；
- KD / NOC2L；
- 真实 scVI encode/decode；
- route-specific DAVF inference；
- token index 与 scVI decoder index 分离；
- E2E CLI direction manifest。

旧 `tests/integration/test_davf_pipeline.py` 全文件没有作为通过依据。该既有兼容测试会进入外部 GeneMapper 等待，本轮手动中止；新增链路测试不依赖该网络路径。

## 7. 当前完成度

已经完成的是工程 E2E 可用性：

```text
真实 context
    → scVI encode
    → route-specific DAVF
    → 方向 evidence
    → 三方 direction gate
    → PerturbGen 双路径 stage plan
    → KO/KD 报告合并
```

尚未完成的是正式生物学验收：

- 当前本地 PerturbGen 正式可运行数据仍不是 `normal/disease + raw counts + 至少 3 donors` 的合规 cohort；
- Datlinger 数据只能作为工程 smoke，不能产生正式效用 PASS；
- KO 当前训练目标数量较少，不能据此宣称广泛泛化；
- DAVF flow loss 不能替代 held-out gene-direction accuracy；
- 正式发布仍需要独立 target、至少 3 个 seed、方向准确率、相关性、稳定性和至少 99 个 null 对照；
- 最终双路径结果必须满足 `source_intervention AND within_state`。

## 8. 风险与运行边界

- 底层 `run_perturbgen_pipeline.py` 是纯 runner，不会自动执行 DAVF gate；正式主线应使用 `run_davf_perturbgen_e2e.py`。
- PerturbGen tokenized 输出由上游固定路径公式管理，多个候选应顺序运行，避免并发写同一 tokenized dataset。
- KO 和 KD 的 latent 数值不能直接比较，也不能拼接后再推理。
- 任何缺失 checkpoint、scVI adapter、embedding asset、gene order 或 context covariate 的情况都应直接失败。
- 真实桥接通过只证明接口和资产契约，不证明最终候选具有生物学效用。

## 9. 相关文件

- [KO/KD DAVF 训练指南](davf_ko_kd_training.md)
- [DAVF × PerturbGen E2E 指南](davf_perturbgen_e2e.md)
- [DAVF × PerturbGen 编排器](../../src/integration/perturbgen/orchestrator.py)
- [E2E CLI](../../scripts/run_davf_perturbgen_e2e.py)
- [KO/KD 报告合并 CLI](../../scripts/merge_davf_perturbgen_reports.py)
- [当前状态](../CURRENT_STATUS.md)

本报告只记录工程实现、真实资产连接和已执行测试，不把 smoke 结果或不完整数据集描述为正式生物学结论。
