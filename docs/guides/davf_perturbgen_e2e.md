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
