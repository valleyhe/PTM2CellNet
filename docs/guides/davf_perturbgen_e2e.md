# DAVF × PerturbGen 串联式 E2E 流程

## 核心原则

KO 和 KD 使用各自的 DAVF checkpoint 与各自的 scVI 坐标系，不能合并 latent
向量。一次候选运行的顺序固定为：

```text
scVI AnnData → scVI.encode → route-specific DAVF
             → gene-expression direction evidence
             → PTM/DAVF/independent-expression 三方 gate
             → PerturbGen source_intervention + within_state
             → 按 Ensembl ID 合并 KO/KD 结果
```

DAVF 的 PerturbGen token 只用于 latent 条件输入；scVI decoder 的目标列只从
当前 adapter 的 `gene_names` 解析。`PTMDirectionMapper.map_ptms()` 仍保留给
历史 API/特征路径；正式方向推理必须使用
`map_intervention_targets(..., "KO"|"KD")`，避免把 PTM 类型先验误当作
checkpoint 干预类型。

## 候选清单

`scripts/run_davf_perturbgen_e2e.py` 接受一个 JSON 文件。每个候选必须指定一个
真实 scVI context 行，不能让脚本猜测 context：

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

`context_h5ad` 的基因顺序和 scVI checkpoint 必须完全一致，并且包含保存模型
要求的协变量；当前 KO/KD 模型要求 `davf_batch`。候选的 symbol/Ensembl pair
还必须同时存在于 DAVF alias asset 和 verified PerturbGen vocabulary。

正式执行 `--run-perturbgen` 时，`stages.tokenise.args.h5ad_path` 必须是已存在的
绝对路径，并在解析后与候选清单的 `context_h5ad` 完全相同；不一致直接硬失败。
tokenise 的 `var_list` 必须按顺序明确声明 cell type、state、donor，且与
`main_pairing_obs`、`time_obs` 及 `[reference_time, disease_state]` 一致。脚本会
在调用外部 PerturbGen runner 前，对同一份完整 context 调用
`prepare_perturbgen_anndata`，验证 raw counts、Ensembl、normal/disease、显式
donor 和至少 3 个共享 donor。Gate-0 未通过时不会进入任何外部 PerturbGen stage。

当前登记的外部 tokeniser 直接消费 `tokenise.args.h5ad_path`；Gate-0 的
`prepare_perturbgen_anndata` 返回的是校验副本。登记源码
`GF_tokenisation.py:223-226,238,301-305` 已静态核对同一原始文件的
counts→`X`→恢复 counts→重算 `n_counts` 接线，prepared copy 的新增 `n_counts` 不需
materialize；原始 tokenise 输入中的 Ensembl ID 必须已经是 canonical 值，脚本会在
进入 stage 前硬失败，不会凭空写入 X/layer 或另造输入文件。本轮未运行真实
tokenisation。

## 只执行真实 DAVF 与方向 gate

```bash
python scripts/run_davf_perturbgen_e2e.py \
  --davf-config configs/davf_ko.yaml \
  --candidate-spec path/to/ko_candidates.json \
  --output outputs/davf_perturbgen/ko_report.json
```

这一步会真实加载当前 checkpoint、scVI 模型和 embedding asset，输出
`schema_version=davf_perturbgen_e2e/v1` 的清单。gate 未通过的候选不会生成
PerturbGen invocation。

## 触发六阶段 PerturbGen

```bash
python scripts/run_davf_perturbgen_e2e.py \
  --davf-config configs/davf_ko.yaml \
  --candidate-spec path/to/ko_candidates.json \
  --perturbgen-config configs/integration/perturbgen.yaml \
  --perturbgen-output-root outputs/perturbgen/e2e \
  --run-perturbgen \
  --output outputs/davf_perturbgen/ko_report.json
```

每个通过 gate 的候选会获得独立的
`<output_root>/<KO|KD>/<Ensembl ID>/` 目录；其中两个 perturb 阶段分别使用
`source_intervention` 和 `within_state`，公共训练阶段仍由既有
`PerturbGenRunner` 管理 GPU 锁、artifact 引用、manifest 和 resume。

`configs/integration/perturbgen.yaml` 当前 `pipeline.random_seed` 为 42。E2E
invocation 会从该 base config 读取并记录该 seed；正式多 seed 仍须显式传入匹配的
`--seeds`，例如 `--seeds 42,43,44`。直接 runner 会硬校验 report 与当前 base
config 的 gene/mode/route/seed/path 绑定；不能依赖旧的默认 seed 0 或把任意通过
report 用于另一份 YAML。

这条命令只代表六阶段执行成功，不自动把缺少 donor/null/FDR 证据的结果标成
生物学 PASS。最终效用判定仍需使用现有 `results.py`、`dual_path.py` 和
`mainline.py`，并满足双路径严格 AND。

## 独立 KO/KD 报告合并

两次运行必须分别使用 `configs/davf_ko.yaml` 与 `configs/davf_kd.yaml`，然后：

```bash
python scripts/merge_davf_perturbgen_reports.py \
  --reports outputs/davf_perturbgen/ko_report.json \
            outputs/davf_perturbgen/kd_report.json \
  --output outputs/davf_perturbgen/ko_kd_merged.json
```

合并键只能是 canonical Ensembl ID。相同 Ensembl ID 的 symbol 冲突、同一路径
重复报告或 route 元数据不一致都会直接失败。

## 当前真实边界

本地已经有 KO/KD DAVF 与 scVI 资产，因此可以验证上述方向桥接。当前本地
PerturbGen 可运行资产仍是 Datlinger 小规模工程 smoke；它不是符合
`normal/disease + >=3 donors + raw counts` 契约的正式效用 cohort。因此正式
PerturbGen 生物学验收前，仍必须准备并通过 `data_prep.py` 的 donor/state
preflight，不能用 scPerturb 的 KO/KD batch 数据替代。
