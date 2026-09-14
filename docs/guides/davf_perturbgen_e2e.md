# DAVF × PerturbGen 串联式 E2E 流程

主线入口是 `PTM site presence → 外部候选方向/位点覆盖 → DAVF 方向证据 → 独立
表达方向 → 三方 gate → PerturbGen invocation → 六阶段运行与双路径评估`。PTM
classifier 只输出位点存在概率；它不从原始 PTM site 自动产生表达方向。

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
      "observed_direction": "down",
      "semantic_context": {
        "context": "normal_to_disease_transition",
        "intervention": "KO",
        "comparison_baseline": "normal_donor_baseline",
        "reference_axis": "disease_minus_normal",
        "research_objective": "reversal",
        "evidence_source": "donor_level_expression_and_davf_decode",
        "cohort": "cohort_manifest_v1"
      }
    }
  ]
}
```

`context_h5ad` 的基因顺序和 scVI checkpoint 必须完全一致，并且包含保存模型
要求的协变量；当前 KO/KD 模型要求 `davf_batch`。候选的 symbol/Ensembl pair
还必须同时存在于 DAVF alias asset 和 verified PerturbGen vocabulary。

公开 candidate spec 的每个候选都必须提供 `semantic_context` 的七个字段：
`context`、`intervention`、`comparison_baseline`、`reference_axis`、
`research_objective`、`evidence_source`、`cohort`。`research_objective` 只能是
`association`、`replication` 或 `reversal`；缺失、空值或非法值会在 formal
`PerturbGenInvocation` 处硬失败。`semantic_context.intervention` 必须与当前
KO/KD route 一致，candidate 与 invocation 的七字段必须一致。示例中的值只是
schema 示例，实际 reference axis、cohort 和证据来源必须由研究设计明确填写，不能
用全局同号/取反代替。

候选中的 `proposed_direction` 是外部用户假设或 `--proposal-direction-map` 的
site-level override；`observed_direction` 若来自 GSE，含义是 donor-level
`disease - normal`。DAVF 的 `predicted_direction` 则来自“干预后 decode - 当前
context decode”。当前 gate 代码只比较三者是否同号，没有对齐参考轴；因此不能把
病程方向当作干预标签或统一取反。正式研究必须显式记录 context、干预、对比基准和
目标（病程关联、复现或逆转），当前 gate 通过仍只是工程准入，不能宣称科学因果
gate 已解决。

正式执行 `--run-perturbgen` 时，`stages.tokenise.args.h5ad_path` 必须是已存在的
绝对路径，并在解析后与候选清单的 `context_h5ad` 完全相同；不一致直接硬失败。
tokenise 的 `var_list` 必须按顺序明确声明 cell type、state、donor，且与
`main_pairing_obs`、`time_obs` 及 `[reference_time, disease_state]` 一致。脚本会
在调用外部 PerturbGen runner 前，对同一份完整 context 调用
`prepare_perturbgen_anndata`，验证 raw counts、Ensembl、normal/disease 与显式
donor。Gate-0 未通过时不会进入任何外部 PerturbGen stage。

队列配对语义由 `--perturbgen-cohort-pairing` 显式声明（L-2026-0914-01）：

- `within_donor`（默认，扰动配对队列）：同一 donor 在两个 state 都有细胞，
  要求 ≥3 个跨态共享 donor；
- `between_donor`（case-control 队列，如 AD 脑队列）：每位 donor 只属于一个
  state，要求两组 donor 不相交（同 donor 双态出现即标签错误硬失败）且每组
  ≥3 个 donor；报告的 `evaluable_donors` 是两组并集，`normal_donors`/
  `disease_donors` 分别记录分组。

首个通过 between_donor Gate-0 的真实队列是
`data/AD/standardized/GSE174367_ad_cohort.h5ad`（61,472 cells × 58,676
canonical ENSG；7 Control + 11 AD donor；由
`scripts/standardize_gse174367_ad_cohort.py` 生成，donor 推导与 ENSG 合并
证据见同目录 provenance JSON；preflight 7/7 细胞类型 PASS）。这只是数据
契约验收，不构成生物学 PASS。

当前登记的外部 tokeniser 直接消费 `tokenise.args.h5ad_path`；Gate-0 的
`prepare_perturbgen_anndata` 返回的是校验副本。登记源码
`GF_tokenisation.py:223-226,238,301-305` 已静态核对同一原始文件的
counts→`X`→恢复 counts→重算 `n_counts` 接线，prepared copy 的新增 `n_counts` 不需
materialize；原始 tokenise 输入中的 Ensembl ID 必须已经是 canonical 值，脚本会在
进入 stage 前硬失败，不会凭空写入 X/layer 或另造输入文件。本轮未运行真实
tokenisation。

E2E 报告中的 `statistical_evidence` 默认写 `status="inconclusive"` 和
`scientific_acceptance=false`；未请求 `--run-perturbgen` 时 reason 为
`perturbgen_not_requested`，请求六阶段但未组装统计时 reason 为
`statistical_evidence_not_assembled`。传入 `--assemble-statistical-evidence`
（要求真实 `--run-perturbgen` 执行）后，E2E 会自动串接现有统计接口：按候选从
primary-mode `within_state` h5ad 提取未扰动质量、以 manifest 绑定的 matched-null
分布组装 formal 评估输入、执行候选 empirical-p 聚合与 BH-FDR、产出双路径 AND
判定并把完整 lineage 写回 `statistical_evidence`。该组装需要显式提供
`--deg-table`（donor/gene_symbol/log2fc/fdr 列）和 `--null-distribution-manifest`
（`perturbgen_null_distribution/v1` 索引），缺 null/质量/seed 覆盖会硬失败或保持
INCONCLUSIVE，绝不产生伪造 p/q。mock、synthetic、smoke 或 bridge 运行不等于
biology PASS。

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
  --seeds 42,43,44 \
  --train-donors D1,D2 \
  --held-out-donors D3,D4,D5 \
  --require-donor-split \
  --frozen-cohort-manifest path/to/frozen-cohort.json \
  --output outputs/davf_perturbgen/ko_report.json
```

E2E 现在对每个 route 公共准备一次：`tokenise/train_mask/train_decoder` 在
`<output_root>/<KO|KD>/_prepare/` 下只执行一次（runner manifest + resume 语义，
重跑传 `--resume` 复用），随后每个通过 gate 的候选只在
`<output_root>/<KO|KD>/<Ensembl ID>/` 下执行自己的
`perturb → export_gene_embeddings → report`，候选计划中的
`@artifact:tokenise/train_*` 引用解析到共享准备产物。报告的
`perturbgen_prepare` 节记录共享准备的 root、阶段与结果；每个
`perturbgen_runs` 条目记录其 `prepare_root`。`PerturbGenRunner` 负责 GPU 锁、
artifact 引用、manifest 和同目录 resume。单路结果只支持对应研究场景，只有两路
都通过才可进入正式 AND 判定，不能解释成普遍治疗疗效。

`configs/integration/perturbgen.yaml` 当前 `pipeline.random_seed` 为 42。E2E
invocation 会从该 base config 读取并记录该 seed；正式多 seed 仍须显式传入匹配的
`--seeds`，例如 `--seeds 42,43,44`。`scripts/run_perturbgen_pipeline.py` CLI 会硬校验 report 与当前 YAML 的
gene/mode/route/seed/path 绑定；不能依赖旧的默认 seed 0 或把任意通过
report 用于另一份 YAML。

训练与 tokenise 的 donor 身份现已可显式绑定。正式 held-out 声明必须同时提供
`--train-donors` 与 `--held-out-donors`（≥2 / ≥3、互斥），可选
`--frozen-cohort-manifest` 按 SHA 逐值核对；缺列表时 `donor_split_status` 为
`unspecified`，不能声称 held-out 细胞未进入 DAVF/PerturbGen 训练。参数绑定不会
替代对 cohort 来源和 DAVF 训练 donor 的独立性审计。匹配 null 的
batch 生成入口是 `scripts/run_matched_null_stages.py`（`--dry-run` 或
`--records-json`）；GPU 执行必须经 Python API 提供 `rescue_extractor`，本模块
不猜测 DEG 列。

这条命令只代表六阶段 stage/manifest 执行成功；`report` 只是 manifest 汇总，不
自动把缺少 donor/null/FDR/未扰动质量或方向轴定义的结果标成生物学 PASS。统计闭环
由 `--assemble-statistical-evidence` 显式触发（见上文 statistical_evidence 段）；
不启用时 `build_dual_path_eval_input.py` 与 `evaluate_perturbgen_dual_path.py`
仍可独立两步调用，但两步调用容易漏接，正式入口应使用 E2E 内置组装。

gate 边界（F-09 的现行约定）：正式 PerturbGen 执行的唯一公开入口是绑定通过
E2E gate report 的 `scripts/run_perturbgen_pipeline.py --e2e-gate-report` 与
orchestrator invocation（`PerturbGenInvocation` 拒绝非 pass 候选）。低层
`PerturbGenRunner.run_pipeline` 只执行 `StagePlan`，属工程执行层，不自行重查
DAVF gate——内部 null/engineering 调用复用该层，但绕过外层 wrapper 直接把
StagePlan 交给 runner 不能视为正式候选执行。该边界尚未下沉到 runner 接口层。

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
