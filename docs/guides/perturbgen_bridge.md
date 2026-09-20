# PerturbGen 桥接指南（DAVF × PerturbGen 双路径整合）

> **文档版本**：v1.8（2026-09-17，补充策略 B 外部证据边界与 APOE 锚点回测结论）
> **权威方案**：[`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](../DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)（v2.0）
> **详细执行方案**：[`docs/guides/davf_ko_kd_training.md`](davf_ko_kd_training.md)
> **状态基线**：[`project_analysis_20260917.md`](../../project_analysis_20260917.md)（本报告按源码、测试和真实资产边界逐项审计；工程契约与科学验收分开记账）

本指南面向需要运行 PerturbGen 训练/扰动链路或 DAVF 嵌入底座迁移的操作者，
给出环境、数据契约、六阶段 pipeline、嵌入资产与评估的入口命令。
**所有命令均为当前仓库真实存在的脚本与参数**（未实现的项明确标注）。

研究主线是 `PTM site presence → 外部提出的候选方向/位点覆盖 → DAVF 方向证据
→ 独立表达方向证据 → 三方 gate → PerturbGen invocation → 双路径效用评估`。
PTM classifier 只判断位点是否存在；`proposed_direction` 必须来自外部假设或
site-level override，不能由原始 PTM site 自动变成干预表达方向。

---

## 1. 架构一页纸

两条工作流（方案 §4.1）：

| 工作流 | 内容 | 入口 |
|---|---|---|
| A：PerturbGen 双路径效用 | 通过三方 gate 的 invocation → 公共数据/模型准备 → `src`/`tgt` 扰动 → rescue/null/donor 统计 → 双路径判定 | `scripts/run_davf_perturbgen_e2e.py` + `scripts/run_perturbgen_pipeline.py` |
| B：DAVF 底座迁移 | PerturbGen encoder ckpt → 静态 gene embedding 资产导出 → schema v2 注入 → 当前 LatentDAVF 重训 → Gate-E | `scripts/export_perturbgen_gene_embeddings.py` + `scripts/train_latent_davf.py` |

主进程与独立环境边界（方案 §4.2，硬约束）：

- 主项目环境（如 `SSH_unit`）：输入契约、路径安全、symbol↔ENSG 解析、配置生成、manifest、stage 调度、输出 schema 校验、统计与报告。
- PerturbGen 独立环境（conda env `perturbgen`，Python 3.11）：官方 tokenisation、masking/count decoder 训练、`src/tgt` 扰动推理、gene embedding 导出。
- **主进程绝不 `import perturbgen`**；所有跨环境调用为参数数组（禁止 `shell=True`），stage 带 timeout、退出码与输出 schema 校验。

`run_davf_perturbgen_e2e.py` 默认只生成 DAVF→三方方向 gate→invocation；只有显式
`--run-perturbgen` 才调用外部六阶段。直接调用
`run_perturbgen_pipeline.py` 的 `perturb` stage 或指定 `--path` 时，必须显式提供
含通过 candidate/invocation 的 `--e2e-gate-report`，缺失或 gate 非 pass 直接失败。
`build_dual_path_eval_input.py` 从成功 stage manifest 组装评估输入；新 null
distribution manifest 按 candidate/path/mode/seed 绑定，不能把 mock 输出当成真实科学闭环。
`scripts/run_perturbgen_pipeline.py` CLI 还会把 report invocation 与当前 base YAML 的 gene、mode、声明的 route/
Ensembl、`pipeline.random_seed` 和授权 path 子集逐项绑定；`PerturbGenInvocation.to_dict()`
中的 `paths` 是授权双路径集合，`perturbgen_config_path` 是原始 base config 路径，
不是 materialized stage YAML。模板 seed 为 42 时，应使用匹配的 report/`--seeds`
（例如 `--seeds 42,43,44`）；seed 不匹配直接失败。

这里的三方方向目前是工程准入检查：代码只比较 proposal、DAVF decode delta 与
observed direction 的同号关系。`observed_direction` 是 donor-level
`disease - normal`，DAVF 方向是“干预后 decode - 当前 context decode”，两者的
context、干预、对比基准和目标（病程关联、复现或逆转）尚未由当前实现统一定义。
因此不能自动取反，也不能把同号 gate 解释为已解决的科学因果 gate；正式研究须把
这些轴和来源隔离写入可追溯资产。

六阶段的当前事实与研究目标必须分开看：

| 阶段 | 当前代码事实 | 研究目标/解读 |
|---|---|---|
| `tokenise`、`train_mask`、`train_decoder` | 自第四轮（2026-09-13）起，E2E 对每条 route 在 `<root>/<route>/_prepare/` 公共执行一次（`orchestrator.build_shared_prepare_plans`）；候选以 `skip_prepare_stages` 复用共享产物，缺失引用硬失败 | 固定 cohort、词表、训练配置和资产版本后只做一次公共准备（已落地） |
| `perturb` | 按 `source_intervention=[src]` 和 `within_state=[tgt]+pert_tps` 生成候选效用运行 | 候选阶段只运行两条研究场景并保留路径身份 |
| `export_gene_embeddings` | 使用配置中的基础 encoder checkpoint；不读取候选阶段训练 checkpoint，也不回灌当前 DAVF | 静态冻结 embedding 资产属于 Workflow B 生命周期 |
| `report` | 汇总 stage manifest 和产物状态 | 仅 manifest 汇总；成功不等于正式生物学 PASS |

跨候选公共准备已由 E2E 落地（2026-09-13 第四轮）：`build_shared_prepare_plans` 在
`<root>/<route>/_prepare/` 公共执行 `tokenise → train_mask → train_decoder`，候选循环经
`resolve_prepare_artifact_references` 把 `@artifact` 引用解析到共享产物（缺失即硬失败），
只执行两路 `perturb/export_gene_embeddings/report`。`--resume` 仍只用于同一输出目录内
恢复既有 stage，不是跨候选复用入口。

## 2. 环境准备（一次性）

```bash
# 1) 独立 Python 3.11 环境（复用已存在的 perturbgen env；绝不触碰共享环境）
bash scripts/setup_perturbgen_env.sh

# 2) 离线 wheel 预下载（网络受限机器转机安装用）
bash scripts/download_perturbgen_wheels.sh

# 3) 环境证据采集（GPU/CUDA/包版本快照，写入 outputs/ 供 manifest 引用）
python scripts/collect_perturbgen_env_evidence.py
```

前提：`ref/Perturbgen-src`（浅克隆）与用户手动放置的 PerturbGen encoder
权重（HuggingFace `lotfollahi-lab/PerturbGen`）。**agent 不得自动拉取大权重**。

### 2.1 运行时环境变量清单（U8，2026-09-16）

`configs/integration/perturbgen.yaml`（formal）与
`configs/integration/perturbgen_local_adaptation.yaml`（local adaptation）的
全部占位符在运行前必须已导出；`config_builder._expand_env` 对未解析占位符
硬失败。变量语义与本地验证值（D3 探针 `outputs/perturbgen/d3_probe_20260916/`
实测通过）如下。

**formal 配置（`perturbgen.yaml`，22 个）**

| 变量 | 语义 | 本地值（已验证） |
|---|---|---|
| `PTM2CELLNET_PERTURBGEN_PYTHON` | 独立环境 python 可执行文件 | `/home/scu/anaconda3/envs/perturbgen/bin/python` |
| `PTM2CELLNET_PERTURBGEN_ENCODER_CKPT` | PerturbGen encoder checkpoint | `/home/scu/PTM2CellNet/perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt` |
| `PTM2CELLNET_PERTURBGEN_TOKEN_DICT` | Geneformer token 词表 pkl | `ref/Perturbgen-src/perturbgen/pp/token_dict_gftokens_gc95M.pkl` |
| `PTM2CELLNET_PERTURBGEN_GENE_MAPPING` | symbol→ENSG 映射 pkl | `ref/Perturbgen-src/perturbgen/pp/ensembl_mapping_dict_gc95M.pkl` |
| `PTM2CELLNET_PERTURBGEN_GENE_MEDIAN` | token 中位数字典 pkl | `ref/Perturbgen-src/perturbgen/pp/gene_median_dict_gftokens_gc95M.pkl` |
| `PTM2CELLNET_PERTURBGEN_INPUT_H5AD` | 正式输入 cohort h5ad（须过 Gate-0） | 待正式候选 context（未冻结） |
| `PTM2CELLNET_PERTURBGEN_DATASET_NAME` | tokenise 数据集名（tokenized_data 子目录） | 待正式冻结 |
| `PTM2CELLNET_PERTURBGEN_CELLTYPE_OBS` / `_STATE_OBS` / `_DONOR_OBS` | tokenise 需要的 obs 列名 | 按正式 context 冻结 |
| `PTM2CELLNET_PERTURBGEN_REFERENCE_STATE` / `_TARGET_STATE` | 参考态/目标态标签 | 按正式队列冻结 |
| `PTM2CELLNET_PERTURBGEN_OUTPUT_ROOT` | 六阶段输出根目录 | 每次运行显式指定 |
| `PTM2CELLNET_PERTURBGEN_ESTIMATED_OUTPUT_BYTES` | 磁盘预算估计（字节） | 按产物规模估计 |
| `PTM2CELLNET_PERTURBGEN_EMBEDDING_VOCAB` / `_TENSOR_KEY` / encoder 相关 | `export_gene_embeddings` 资产参数 | 按导出资产冻结 |
| `PTM2CELLNET_PERTURBGEN_PERT_TP` / `_PERTURBATION_SEQUENCE` / `_TARGET_GENE` | perturb 阶段干预参数 | 按候选 spec 冻结 |

**local adaptation 配置（`perturbgen_local_adaptation.yaml`，9 个）**：`PTM2CELLNET_PERTURBGEN_PYTHON`（同上）与 8 个 `PTM2CELLNET_PERTURBGEN_LOCAL_*`：
`_INPUT_H5AD`（m0 smoke 子集 `ref/Perturbgen-src/data/perturbgen_m0_smoke/datlinger2021_m0smoke.h5ad`）、`_DATASET`（tokenized 数据集名）、`_ENCODER_CKPT`、`_TOKEN_DICT`、`_GENE_MAPPING`、`_GENE_MEDIAN`（均同 formal 路径）、`_TARGETS`（`ref/Perturbgen-src/data/perturbgen_m0_smoke/genes_to_include.csv`）、`_OUTPUT_ROOT`。

**P40 注意**：`train_mask`/`train_decoder` 的 `torch.compile`（triton）要求
CUDA Capability ≥ 7.0；Tesla P40（6.1）必须 `export TORCHDYNAMO_DISABLE=1`
（eager 模式，探针实测 8.7s/43.0s/37.4s、峰值 4,359 MiB）。

## 3. 数据契约（Gate-0，先于一切训练）

donor cohort 硬要求（方案 §4.6-1；lessons.md L-2026-0822-06、L-2026-0914-01）：

1. 显式 `donor`/`patient` 列（`sample`/`batch`/`replicate`/CRISPR control 不算）；
2. normal/disease 配对、raw counts、Ensembl ID（ENSG）、`cell_type`/`state` 元数据；
3. 目标 cell type 每状态 ≥3 个可评估 donor，配对语义二选一：
   `within_donor`（扰动配对，≥3 个跨态共享 donor，默认）或
   `between_donor`（case-control，两组 donor 不相交且各 ≥3；AD 脑队列采用）。

```bash
# 审计本地 h5ad 是否有合规候选（结果写 outputs/perturbgen/spike/<date>_donor_audit/evidence.json）
python scripts/audit_perturbgen_cohort.py
# 审计 data/AD 四个 GEO 队列的 Gate-0 差距（donor/condition/Ensembl 逐项）
python scripts/audit_ad_cohort_gate0.py
# 标准化 GSE174367 并跑 between_donor Gate-0 preflight（全部细胞类型）
python scripts/standardize_gse174367_ad_cohort.py
```

**历史状态（2026-09-10）**：30 文件审计 0 合规候选 —— M0⑥ 是全链外部
数据硬阻断（U-01）。

**本轮状态（2026-09-13）**：重新登记的
`outputs/perturbgen/spike/20260913_donor_audit/evidence.json` 审计 30 个文件、26
个可读、0 个合规候选；未以旧 2026-09-03 evidence 冒充本轮日期。Gate-0 未过时，
M4 重训/M6/Gate-E 仍按方案 §7.3 挂起，不得跳过。训练输入 schema/asset 通过也不
证明 held-out DAVF biology，因为来源隔离和训练/E2E donor split 仍须由 manifest
逐值追溯。正式验收还必须同时满足真实 `normal/disease` raw counts、显式 donor、
至少 3 个可评估 donor（配对语义见上）、canonical Ensembl、scVI gene order、冻结
embedding 与 manifest；smoke、synthetic 和 bridge 通过都不算生物学 PASS。

**AD 队列状态（2026-09-14，L-2026-0914-01；同日 F-14 修复后重审）**：审计脚本已
改为双 pairing 口径，重跑 verdict 为
`GATE0_UNLOCKED_FOR_BETWEEN_DONOR_GSE174367_ONLY`
（`outputs/perturbgen/spike/20260914_ad_cohort_audit/evidence.json`）；
Gate-0 新增 `between_donor` 配对后，GSE174367 已标准化为
`data/AD/standardized/GSE174367_ad_cohort.h5ad` 并以 between_donor preflight
7/7 细胞类型 PASS
（`outputs/perturbgen/spike/20260914_gse174367_gate0/evidence.json`）。
GSE157827/GSE188545/GSE147528 仍缺 donor/cell 注释或 cell calling，未合并。
该 preflight 是数据契约验收，不是生物学 PASS。

**M6 冻结状态（2026-09-14，第八轮，L-2026-0914-03）**：GSE174367 已完成
between_donor M6 数据契约冻结（EX 细胞类型，目录
`outputs/perturbgen/frozen/20260914_gse174367_ex/`）：

| 资产 | 内容 | 校验 |
|---|---|---|
| `manifest.json` | `ptm2cellnet.frozen-cohort/v1`，pairing=between_donor，EX，sha256 绑定 cohort h5ad | `load_frozen_manifest` 加载即重验 sha256 |
| donor split | train 12（Control 4 + AD 8）/ held-out 6（Control 3 + AD 3，两组各 ≥3），分层 seed=2 抽样且四组均两性平衡，并集覆盖全部 18 donor | `_validate_frozen_cohort_asset` 真实数据 PASS（6,369 cells，无 issue） |
| `candidates.csv` | APP/PSEN1/BACE1/MAPT/APOE（KO，AD 三通路外部假设）；ENSG 从 cohort var 查证，token index 在 embedding asset 词表全部有效 | `provenance.json` 记录双重查证 |
| `acceptance_plan.json` | 5 候选 × 2 path × 3 seed × 3 mode = 90 runs 矩阵 | `donor_leakage: clear` |
| `null_selection/*.json` | 每候选 99 个表达特征匹配 null（EX 子集 mean/detection + embedding token_rank；fc 显式标记 `unavailable/default`，DEG 表就绪后可重生成启用 fc 特征） | `perturbgen_null_selection/v1` |
| `tests/real_assets/test_real_frozen_cohort.py` | 冻结契约漂移守护（加载重验 + 资产校验 + between_donor 不相交断言） | gate 开启时真实跑 18.25s PASS |

**GPU 执行 runbook（脱离沙箱，按序执行）**：

```bash
# 0. 前置校验（CPU，~18s）：确认冻结契约与资产未漂移
PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 pytest tests/real_assets/test_real_frozen_cohort.py -v

# 1. 准备 candidate_spec（研究输入，不可自动生成）：每候选需真实文献 PTM site
#    锚点（position>=1 + ptm_type，契约必填）、外部方向假设 proposed_direction、
#    donor-level disease−normal DEG 证据（observed_log2fc/fdr/direction）与
#    semantic_context 七字段；context_h5ad 指向标准化 cohort，context_cell_index
#    必须落在 EX。DEG 表列名与 E2E --deg-*-column 参数一致。
#    上游 PTM activity → AD 交集主线可用
#    scripts/build_celltype_candidate_specs.py 从交集产物批量生成该 spec
#    （同一 candidate-spec/v1 契约），见 docs/guides/ptm_activity_pipeline.md §6；
#    source_proposals 表中的位点锚点与方向假设仍属研究输入，不自动发明。

# 2. 先只执行六阶段（GPU；此时不要传 --assemble-statistical-evidence、
#    --deg-table 或 --null-distribution-manifest；train/held-out 列表与冻结
#    manifest 逐字一致，bind_frozen_donor_split 会硬校验）
python scripts/run_davf_perturbgen_e2e.py \
  --davf-config configs/davf_ko.yaml \
  --candidate-spec <candidate_spec.json> \
  --perturbgen-config configs/integration/perturbgen.yaml \
  --output outputs/davf_perturbgen/gse174367_ex/e2e_report.json \
  --run-perturbgen \
  --perturbgen-cohort-pairing between_donor \
  --seeds 0,1,2 \
  --sensitivity-modes pad,delete \
  --frozen-cohort-manifest outputs/perturbgen/frozen/20260914_gse174367_ex/manifest.json \
  --train-donors "Sample-52,Sample-58,Sample-66,Sample-82,Sample-17,Sample-27,Sample-33,Sample-43,Sample-45,Sample-46,Sample-47,Sample-50" \
  --held-out-donors "Sample-100,Sample-90,Sample-96,Sample-19,Sample-22,Sample-37"

# 3. 在第 2 步通过后，先完成真实 matched-null 批跑（GPU；CLI 只提供 --dry-run
#    计划，执行走 Python API）。下面是一个 candidate × path × mode × seed 组合的
#    真实绑定入口；研究代码必须重复覆盖全部 5 × 2 × 3 × 3 组合，并在外部真实数据
#    绑定完成后写出实际的 null distribution manifest。rescue 提取器
#    build_stage_rescue_extractor 复用与 candidate path 提取完全相同的
#    pred_counts/X donor 聚合与 held-out DEG signature 打分（排除被扰动的 null
#    基因本身），并从 stage manifest 登记 result h5ad sha256；DEG 表与
#    var[var_gene_column] 必须使用同一基因命名空间，不足 3 个可评估 donor 硬失败。
#    不得用 synthetic rescue 分数代替真实输出。
python - <<'EOF'
from src.integration.perturbgen.null_generation import run_matched_null_stages
from src.integration.perturbgen.null_rescue import build_stage_rescue_extractor
from src.integration.perturbgen.replay_evaluation import load_deg_table

rescue_extractor = build_stage_rescue_extractor(
    load_deg_table("outputs/davf_perturbgen/gse174367_ex/deg_table.tsv"),
    donor_obs_column="donor",
    var_gene_column="__index__",
)

result = run_matched_null_stages(
    selection_manifest="outputs/perturbgen/frozen/20260914_gse174367_ex/null_selection/APP.json",
    base_config="configs/integration/perturbgen.yaml",
    e2e_gate_report="outputs/davf_perturbgen/gse174367_ex/e2e_report.json",
    path="source_intervention", mode="mask", seed=0,
    output_root="outputs/perturbgen/nulls/APP",
    rescue_extractor=rescue_extractor,
    output_path="outputs/perturbgen/nulls/APP/source_intervention_mask_seed0.json",
)
EOF

# 4. 全部 matched-null 真实结果和实际 <null-index.json> 就绪后，用同一套 E2E
#    参数加 --resume 复用已完成的六阶段，只组装统计；不要删除或改换 output root。
python scripts/run_davf_perturbgen_e2e.py \
  --davf-config configs/davf_ko.yaml \
  --candidate-spec <candidate_spec.json> \
  --perturbgen-config configs/integration/perturbgen.yaml \
  --output outputs/davf_perturbgen/gse174367_ex/e2e_report.json \
  --run-perturbgen \
  --resume \
  --perturbgen-cohort-pairing between_donor \
  --seeds 0,1,2 \
  --sensitivity-modes pad,delete \
  --frozen-cohort-manifest outputs/perturbgen/frozen/20260914_gse174367_ex/manifest.json \
  --train-donors "Sample-52,Sample-58,Sample-66,Sample-82,Sample-17,Sample-27,Sample-33,Sample-43,Sample-45,Sample-46,Sample-47,Sample-50" \
  --held-out-donors "Sample-100,Sample-90,Sample-96,Sample-19,Sample-22,Sample-37" \
  --assemble-statistical-evidence \
  --deg-table <deg.csv> \
  --null-distribution-manifest <null-index.json>

# 5. 对第 4 步实际组装出的统计产物做独立冻结验收复算。
#    eval_input/report_manifest 从 E2E 报告读取，避免猜测统计目录或文件名。
E2E_REPORT=outputs/davf_perturbgen/gse174367_ex/e2e_report.json
FROZEN_MANIFEST=outputs/perturbgen/frozen/20260914_gse174367_ex/manifest.json
VERIFY_OUTPUT=outputs/davf_perturbgen/gse174367_ex/frozen_verify.json
EVAL_INPUT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["statistical_evidence"]["eval_input"])' "$E2E_REPORT")"
REPORT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["statistical_evidence"]["report_manifest"])' "$E2E_REPORT")"
python scripts/run_frozen_acceptance.py \
  --verify \
  --manifest "$FROZEN_MANIFEST" \
  --eval-input "$EVAL_INPUT" \
  --report-manifest "$REPORT_MANIFEST" \
  --output "$VERIFY_OUTPUT"
```

第 2 步的 `--perturbgen-config`、`--seeds 0,1,2`、`--sensitivity-modes pad,delete`
与冻结矩阵一致：5 候选 × 2 path × 3 seed ×（`mask` 主模式 + `pad`/`delete`
敏感性模式）= 90 runs。第 2 步只产生 gate/E2E stage 报告，不组装统计；第 3 步
完成真实 null 后，第 4 步以 `--resume` 复用既有 stage manifest，并首次传入
`--assemble-statistical-evidence --deg-table --null-distribution-manifest`。由于
`_resolve_statistical_evidence_inputs` 和 loader 要求 null manifest 已存在，不能把
统计参数放在第 2 步。第 5 步的 `--verify` 只在第 4 步报告的
`statistical_evidence.status=assembled` 且已写出 `eval_input`/`report_manifest` 后
执行；缺失或篡改资产、计划覆盖不足、replay 不重现或独立 h5ad 未重算均返回非零，
不能写成冻结验收 PASS。正式 verdict 仍只表示计算效用/方向筛选，不构成治疗或临床
因果结论。

## 4. 六阶段 pipeline（工作流 A）

训练-only 规划可以按实际 `--stages` 语法跳过 `perturb`，因此不需要 gate report：

```bash
python scripts/run_perturbgen_pipeline.py \
  --config configs/integration/perturbgen.yaml \
  --stages tokenise train_mask train_decoder \
  --dry-run \
  --gpu-lock-file /tmp/pg.gpu.lock
```

这只表示公共准备阶段的计划/执行；不产生候选效用，也不能替代通过 gate 的
invocation。`--path` 即使只想规划路径也会触发 gate-report 要求。

```bash
python scripts/run_perturbgen_pipeline.py \
  --config configs/integration/perturbgen.yaml \
  --stages tokenise train_mask train_decoder perturb export_gene_embeddings report \
  --path both \
  --e2e-gate-report outputs/davf_perturbgen/ko_report.json \
  --dry-run \
  --gpu-lock-file /tmp/pg.gpu.lock
```

- stage 顺序固定：`tokenise → train_mask → train_decoder → perturb → export_gene_embeddings → report`；
- 选中 `perturb` 或指定 `--path` 时必须提供真实存在且包含通过候选/invocation 的 DAVF E2E gate report；训练-only 阶段无需该参数；
- `source_intervention` 只表示在状态转移前对 `src` 施加扰动，`within_state` 表示在目标状态对 `tgt` 施加扰动并使用 `pert_tps`；两条路径的正式效用结论必须同时成立。单路结果只适用于对应研究场景，不能称为正式双路径 PASS 或普遍治疗疗效；
- 产物路径按上游公式 + 唯一 glob 解析后写入 manifest（路径 + hash），**禁止
  latest-mtime 猜测**；零匹配/多匹配/hash 变化直接失败（lessons.md L-2026-0822-04）；
- N-05 评估组装会从 runner 的 `outputs` 中按 `artifacts.result_h5ad` 的实际路径复制
  已登记 `sha256` 到 `h5ad_provenance`；assembler 不新算、不猜 h5ad hash；
- 上例用 `--dry-run` 只打印计划；正式续跑时移除 `--dry-run`，需要从 manifest
  继续时再添加 `--resume`。

`PerturbGenRunner` 本身只执行 `StagePlan`，不自校验 DAVF gate。正式候选入口应先
由 E2E 产生通过的 invocation；直接 pipeline CLI 只有在选中 `perturb` 或 `--path`
时才按当前 report binding 校验 gate。把任意 StagePlan 交给 runner 不能当作科学
准入流程。formal invocation wrapper（`run_perturbgen_pipeline.py
--e2e-gate-report` 与 orchestrator `PerturbGenInvocation`）因此是唯一公开正式
执行入口；runner 的 `StagePlan` 层仅供内部 null/engineering 复用，该边界尚未
下沉到 runner 接口层（F-09 现状）。

## 5. 嵌入资产导出与 DAVF 注入（工作流 B）

```bash
# 参数模板：执行前将尖括号替换为真实路径；tensor-key 必填，工具从不猜测 checkpoint 布局
python scripts/export_perturbgen_gene_embeddings.py \
    --checkpoint <encoder.ckpt> \
    --tensor-key <精确的 embedding 参数键> \
    --vocabulary <gene_id→row 的 JSON/pickle> \
    --output-dir outputs/perturbgen/embedding_asset/
```

产出 `gene_embeddings.safetensors + vocabulary.json + manifest.json`。其中嵌入资产
`manifest.json` 使用 `schema_version: 1`（含 sha256 与维度）；DAVF 配置文件另使用
配置 schema v2。主环境加载零 PerturbGen 依赖。该导出属于 Workflow B：资产从基础
encoder checkpoint 静态生成并冻结；六阶段中的 `export_gene_embeddings` 不消费
候选结果或阶段训练 checkpoint，也不把导出结果回灌正在运行的 DAVF。

`finetune_davf_e2e.py` 只训练下游 PTM2CellNet 分类头，不能生成正式方向
checkpoint。当前 DAVF 必须用 flow-matching 入口重训：

```bash
python scripts/train_latent_davf.py \
  --train-data <latent-pair-train.npz> \
  --val-data <latent-pair-val.npz> \
  --scvi-model checkpoints/scvi/ibd_norman_model \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --intervention-type <KO|KD|OE> \
  --output checkpoints/davf/latent_davf_perturbgen_4018 \
  --train-donors <d1,d2> \
  --held-out-donors <d3,d4,d5> \
  --require-donor-split \
  --frozen-cohort-manifest path/to/frozen-cohort.json
```

上面的 donor 参数是正式 held-out 训练的要求；没有真实、可追溯的 donor split 时，
只能把该命令作为工程 smoke，并在 `training_metrics.json` 中接受
`donor_split_status=unspecified`，不能声称训练/验证未泄漏。`OE` 可用于独立模型
训练，但当前串联 E2E orchestrator 的正式 route 是 `KO`/`KD`。

`latent-pair-*.npz` 必须包含 `metadata_json` 以及
`z_0[N,64]`、`z_1[N,64]`、`gene_ids[N,K]`、`directions[N,K]`、
`attention_mask[N,K]`。`metadata_json` 必须记录数据 schema、生成所用 scVI
模型路径与完整有序 `gene_names`，以及 PerturbGen asset 路径、词表大小、维度和
manifest；训练器会将这些字段与命令行传入的实时 adapter/asset 逐项比对。
其中 `gene_ids` 只能是 PerturbGen asset 的 token row。缺少真实 latent pair 或
方向标签时，命令应失败，不能把 gene-space delta 表自动改名后训练。

`metadata_json` 的最小结构如下（`gene_names` 必须是完整的 4018 项，不能用
PerturbGen token 列表代替）：

```json
{
  "schema_version": "ptm2cellnet.latent-davf-pairs.v1",
  "scvi": {
    "model_path": "/absolute/path/to/checkpoints/scvi/ibd_norman_model",
    "latent_dim": 64,
    "num_genes": 4018,
    "gene_names": ["...完整的 scVI decoder gene order..."]
  },
  "embedding_asset": {
    "path": "/absolute/path/to/outputs/perturbgen/embedding_asset_20260822",
    "vocab_size": 18967,
    "embedding_dim": 768,
    "manifest": {"...": "与 asset/manifest.json 完全相同"}
  }
}
```

注入链路：`src/models/davf_inference.py`（`embedding_asset_path` 字段 +
sha256/schema 校验）→ `LatentDAVF(pretrained_gene_embeddings=...)`；
旧 `geneformer_path` 配置在 `architectures.py` 迁移闸门直接 ValueError。
**Gate-E 未过前不删除 Geneformer 主链（M7）**。

### 5.1 DAVF 与 scVI 的连接契约（2026-09-02）

当前本地 `checkpoints/scvi/ibd_norman_model` 的真实 schema 是
`4018 genes × 64 latent`，并使用 `batch` / `dataset` 协变量。连接时必须使用
`ScVIAdapter.from_trained_model()`，再调用
`DAVFInferenceModule.load_scvi_adapter(adata)` 或
`bind_scvi_adapter(adapter)`；适配器会严格检查 latent 维度、decoder 基因数和
AnnData 的基因顺序，并从 AnnData 恢复 decoder 协变量。

```python
adapter = ScVIAdapter.from_trained_model("checkpoints/scvi/ibd_norman_model")
z_0 = adapter.encode(adata)       # [B, 64]，同时保存 batch/dataset 上下文
expression = adapter.decode(z_0)  # [B, 4018]
```

DAVF 配置中的 `num_genes` 指 scVI decoder 的输出维度；PerturbGen asset 的
Ensembl/token vocabulary 是另一套输入 token 索引，不能直接当作 scVI 的 gene
index。正式 checkpoint 使用 `schema_version=2`，并记录当前 `LatentDAVF` 完整
state dict、asset manifest 和 scVI gene vocabulary provenance。运行时的目标 gene
index 始终由 `adapter.gene_names` 解析；checkpoint 中的 gene vocabulary 只用于
一致性核验，不能充当索引来源。

当前旧 `latent_davf_ibd_norman` 和 `model_a_4018` checkpoint 仍是 legacy/旧
`delta_mlp` 架构，没有当前 `LatentDAVF.predict()`，只能用于兼容 feature path，
不能生成正式 DAVF direction evidence。新的 current checkpoint 已由 Norman latent
pairs 重训练得到；可用以下命令验证新旧权重：

```bash
python scripts/validate_latent_davf_checkpoint.py \
  --checkpoint checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt \
  --scvi-model checkpoints/scvi/ibd_norman_model \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822
```

`gene_names_path` 若配置为 Norman 的两列 `Ensembl ID<TAB>gene symbol` 表，
DAVF mapper 会把 API symbol 解析成 PerturbGen token；这张表绝不用于生成 scVI
decoder index，后者仍由 `adapter.gene_names` 解析。重复 symbol 对应多个 ENSG
时只屏蔽该 symbol，不选择任意一个。

## 6. 评估、报告与发布证据

```bash
# 双路径评估（rescue/null/FDR/AND 判定；formal 由输入中的 empirical null runs 聚合 q）
python scripts/evaluate_perturbgen_dual_path.py --input-json <candidates.json> --output-dir <outdir>

# 从 DAVF E2E 报告和 stage manifest 生成评估输入；正式 manifest 每个
# candidate/path/mode/seed 必须有至少 99 个有限 null 值
python scripts/build_dual_path_eval_input.py \
    --e2e-report <e2e-report.json> --output <eval-input.json> \
    --deg-table <deg.csv> --null-distribution-manifest <null-index.json> \
    --unperturbed-quality-status <pass|fail|inconclusive> \
    --evaluation-mode engineering \
    --uniform-candidate-pvalue <p>

# 正式评估输入：不写 candidate p；最终 evaluator 从每条 path/mode/seed 的 empirical null 聚合 q
python scripts/build_dual_path_eval_input.py \
    --e2e-report <e2e-report.json> --output <eval-input.json> \
    --deg-table <deg.csv> --null-distribution-manifest <null-index.json> \
    --evaluation-mode formal \
    --unperturbed-quality <unperturbed_quality.json>

# 匹配 null 的 batch 计划 / 已提取记录汇总（不伪造 candidate invocation）
python scripts/run_matched_null_stages.py \
    --selection-manifest <selection.json> --e2e-report <e2e-report.json> \
    --path source_intervention --mode mask --seed 0 \
    --output-root <null-root> --output <plan-or-distribution.json> --dry-run

# 只 benchmark 公共准备阶段（候选 perturb 必须由已有 gate invocation 入口触发）
python scripts/benchmark_perturbgen.py \
    --config path/to/perturbgen.yaml \
    --stages tokenise train_mask train_decoder \
    --fixture-type engineering --iterations 1

# Gate-4 发布证据校验（单次运行自洽，不接受多次残缺 benchmark 的并集）
python scripts/check_perturbgen_release_evidence.py --evidence <evidence.json> [--benchmark-json <bench.json>] [--mode formal|smoke]

# 统一 formal Workflow A 验收编排（Gate-4/5）：一次消费 E2E gate report、
# frozen cohort manifest、matched-null distribution manifest、未扰动质量与
# dual-path eval input/report manifest，输出单一 verdict（pass/fail/blocked）
# 与缺失项清单；blocked 表示输入尚未提供，绝不静默跳过或降级为 pass。
# 退出码：0=pass，1=fail，2=blocked。质量可由 --quality-json 显式给出，
# 或省略并从 --eval-input 候选的 unperturbed_quality 字段读取。
python scripts/verify_formal_workflow_a.py \
    --e2e-report <e2e-report.json> \
    --frozen-manifest <frozen-manifest.json> \
    --null-distribution-manifest <null-distribution.json> \
    --eval-input <dual-path-eval-input.json> \
    --report-manifest <dual-path-report-manifest.json> \
    --output <formal-verification.json>
```

`engineering` 入口允许 uniform 或外部表格 p，只能标记 synthetic；`formal` 入口
要求 `extract_unperturbed_quality_from_h5ad` 的 JSON，并由
`evaluate_perturbgen_dual_path.py` 对每条路径/seed 的有限 empirical p 使用
`conservative_max_required_runs` 聚合，再做候选层 BH-FDR。E2E 已支持
`--assemble-statistical-evidence --deg-table ... --null-distribution-manifest ...`
在六阶段完成后自动串接这条统计闭环（质量提取、formal 评估输入、empirical p 聚
合、BH-FDR 与双路径 AND，lineage 写回 E2E report 的 `statistical_evidence` 节）；
本页的两步 CLI 仍可独立使用，但正式入口应优先使用 E2E 内置组装，避免两步调用
漏接。评估相关的候选语义上下文（`SemanticContext` 七字段）与 E2E 候选清单相同：
`context`、`intervention`、`comparison_baseline`、`reference_axis`、
`research_objective`（限 `association`/`replication`/`reversal`）、
`evidence_source`、`cohort`，缺字段或非法值在 invocation 处硬失败。

判定口径（lessons.md L-2026-0821-01）：正式双路径 **AND** 标准——`src` 与 `tgt`
两路 rescue 均稳定为正（排除目标基因本身、≥3 donor 方向一致、跨要求的 seed 复核）
才进实验验证候选清单。模式覆盖按当前 route-specific `dual_path.py`：
`observed_direction=up` 且 route=KO 时检查 `mask`、`pad`、`delete`；up 且 route=KD
时只检查 `mask`；`observed_direction=down` 时使用 `overexpress`。这些是路径内的
方向/敏感性检查，不新增路径，也不把 mode 集合扩展成额外的双路径 AND 场景。
`--uniform-candidate-pvalue` 与手填
`--unperturbed-quality-status` 只能用于 `evaluation_mode=engineering`；报告会
标记 `evidence_class=synthetic` 且 `scientific_acceptance=false`。formal 模式
要求 `extract_unperturbed_quality_from_h5ad` 产物，以及
`conservative_max_required_runs` 聚合后的 empirical p。pathway/GSEA 是次级证据，
不写入 dual-path 硬 PASS；单路通过只支持相应 path 的研究场景。

## 6a. 外部扰动证据源（LPM，历史/关闭）

> **状态：已关闭，不是当前执行路径。** 2026-09-17 对 perturblib 默认 K562
> essentialome（2,285 行）和 GWPS 词表的直测推翻了早期 5/5 覆盖推断；当前外部证据
> 源切换到 §6b 的 GEARS + Geneformer。以下命令仅保留作历史实验记录，不能作为
> 当前候选证据或 lineage 输入。

历史背景与选型（2026-09-17 早期实测）：四个候选模型中，STATE 的遗传扰动词表
（State-Replogle-Filtered，2,024 个）不含任何 AD 候选（5/5 MISS，实测）；
GEARS GO 通道与 Geneformer 全部可查但分别是外推/反事实语义；**LPM
（[perturblib](https://github.com/perturblib/perturblib)，Apache-2.0）的
Replogle CRISPRi 通道对 5 候选全部 in-vocab**（推断链：perturblib 加载原始
未过滤 figshare 数据 + Replogle 全表达基因文库设计 + 本地 Frangieh K562
表达实测 5/5 阳性）。当时决策为**单源 LPM 最简起步**，随后被 §6b 的策略 B
取代；Geneformer 作为条件触发的第二源、GEARS/STATE 暂不投入均为历史状态。

### 环境搭建（独立环境，不进 requirements-core）

```bash
python -m venv ~/.venvs/perturblib && source ~/.venvs/perturblib/bin/activate
git clone https://github.com/perturblib/perturblib && cd perturblib
pip install poetry && poetry install          # 或 pip install -e .
```

注意：perturblib 不在 PyPI；其 Replogle 数据从 figshare 下载，本机出口对
figshare TLS 不稳定（lessons L-2026-0916-03）——首次下载数据时如遇 SSL/202
排队，需走代理或镜像下载后放入 perturblib 缓存目录。

### 训练（一次性，按论文配置 5 seeds）

```bash
python -m perturb_gym.training train_from_config_file \
    --config_file_id_or_path=replogle_k562_paper_lpm
```

产物为 5 个 seed 的 checkpoint 目录；记录 perturblib git commit。

### 推理与冻结资产导出（外部环境）

```bash
python scripts/run_lpm_predictions.py \
  --candidates-tsv candidates.tsv \            # ensembl_id/gene_symbol 两列
  --trained-model-dir <5-seed-checkpoint 目录> \
  --context HumanCellLine_K562_10xChromium3-scRNA-seq_Replogle22 \
  --perturbation-semantics crispri_kd \
  --output-h5ad outputs/external_evidence/lpm_predictions.h5ad \
  --manifest-output outputs/external_evidence/lpm_predictions.manifest.json \
  --perturblib-commit <commit>
```

输出 manifest（`ptm2cellnet.lpm-prediction/v1`）绑定 h5ad sha256、训练
配置、context、seeds 与聚合语义。候选不在 LPM 扰动词表时硬失败并列出缺失
清单。首次运行时如 perturblib 的 PlibData 布局与脚本假设不符，按其教程
（`docs/source/notebook_tutorials/01_data.ipynb`）适配构造部分——这是外部
环境适配点，不是主环境契约变更。

### 主环境消费（校验 + evidence payload）

```bash
python scripts/assemble_external_evidence.py \
  --prediction-manifest outputs/external_evidence/lpm_predictions.manifest.json \
  --candidates-tsv candidates.tsv \
  --output outputs/external_evidence/external_evidence.json
```

契约（[`src/integration/perturbgen/external_perturbation_evidence.py`](../../src/integration/perturbgen/external_perturbation_evidence.py)）：
readout 词表必须 canonical Ensembl；候选重复、非 finite delta、语义枚举外
值（`token_mask_ko` 被显式拒绝）、hash 漂移、context 不一致全部硬失败；候选
自身 readout 行缺失时 `self_delta/predicted_direction` 记 None，不零填。

### 边界（与方案 §5.5/§8.6 一致）

- 证据语义是 **K562 基线 context 上的 CRISPRi 扰动外推**，不是疾病 context
  干预；`crispri_kd` 近似但不等于主线 `token_mask_ko`，分字段记录、永不与
  DAVF/PerturbGen 证据合并，不构成 pass/fail、因果验证或 biology PASS。
- **Geneformer 第二源触发条件**（满足其一即启动接入）：① 出现
  `observed_direction=down` 的 pass 候选（OE 路径，LPM 词表仅 PSEN1 有 OE）；
  ② APOE 锚点回测显示 LPM 方向可疑。触发前不搭建。
- **先证后用**：LPM 资产正式进入任何 gate/lineage 消费前，必须先过 APOE
  锚点回测（本地 Frangieh 真实 KO 方向对照 + mean-shift baseline 对照）。
- MAPT 注意：K562 检出率仅 0.6%，其 LPM 训练信号可能接近零——MAPT 的方向
  证据使用时要携带该前缀评估。

## 6b. 外部扰动证据源（当前策略 B：GEARS + Geneformer，2026-09-17 起）

2026-09-17 词表实测定案（lessons L-2026-0917-02/03）：STATE 遗传词表（2,024）
与 perturblib 默认 K562 context（essentialome 2,285）均不含任何 AD 候选；
GWPS context 覆盖 3/5（APOE/MAPT/PSEN1）但训练数据 65.83GB 超本机磁盘。
**LPM/STATE 路径关闭，当前证据源为策略 B 双源**。两源资产虽通过
`external-perturbation-prediction/v1` 输入契约，但 APOE 预注册锚点回测为
`verdict=fail`（top-100 方向一致率 0.25、Spearman -0.0428），因此两源均**不接入
主线 lineage**，也不构成 biology PASS：

| 源 | 通道 | 候选覆盖 | evidence_kind | 语义边界 |
|---|---|---|---|---|
| [GEARS](https://github.com/snap-stanford/GEARS)（cell-gears，MIT） | GO 图谱 unseen 查询（gene2go 5/5，GO terms 45–147） | 5/5 | `go_extrapolation` | 训练模态混合的 GO 外推，K562 基线 context |
| [Geneformer](https://huggingface.co/ctheodoris/Geneformer)（V2-104M，MIT） | 疾病细胞 in-silico perturbation（token 词表 5/5） | 5/5 | `network_counterfactual` | 掩码基因网络反事实，非训练扰动响应 |

### 环境（复用 perturblib conda env 作外部模型环境）

```bash
conda activate perturblib
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124  # P40 需 cu124，勿装 cu13
pip install cell-gears geneformer
```

本地资产：GEARS Norman 语料与 GO 图谱在 `data/raw/gears/`；Geneformer V2-104M
权重与字典在 `checkpoints/geneformer/`（`legacy_v1_davf_base/` 是 2026-08 已废弃
的 DAVF 嵌入底座遗留，勿复用）；symbol→Ensembl 映射表
`data/processed/gene_symbol_ensembl_map.tsv`（自 GSE174367 var 生成，58,676 行）。

### GEARS 训练与 unseen 预测（外部环境）

```bash
python scripts/run_gears_predictions.py \
  --candidates-tsv candidates.tsv \
  --gears-data-dir data/raw/gears/norman \
  [--model-ckpt <ckpt> | --train-epochs 10] \
  --output-h5ad outputs/external_evidence/gears_predictions.h5ad \
  --manifest-output outputs/external_evidence/gears_predictions.manifest.json
```

### Geneformer ISP（疾病细胞反事实，外部环境）

```bash
python scripts/run_geneformer_isp.py \
  --candidates-tsv candidates.tsv \
  --input-h5ad data/AD/standardized/GSE174367_ad_cohort.h5ad \
  --cell-type-obs-value EX \
  --model-dir checkpoints/geneformer \
  --median-file checkpoints/geneformer/gene_median_dictionary_gc104M.pkl \
  --token-dict checkpoints/geneformer/token_dictionary_gc104M.pkl \
  --perturb-mode delete \
  --output-h5ad outputs/external_evidence/geneformer_isp_EX.h5ad \
  --manifest-output outputs/external_evidence/geneformer_isp_EX.manifest.json
```

两源资产同为 `external-perturbation-prediction/v1`（candidates × canonical-Ensembl
readouts delta 矩阵），主环境消费与 §6a 相同（`assemble_external_evidence.py`）。
脚本对 cell-gears/geneformer 具体版本 API 的假设在首跑时可能需要适配（报错信息
指向其官方文档），属外部环境适配点，不是主环境契约变更。

**先证后用不变**：两源资产正式进入任何 gate/lineage 消费前，必须先过 APOE 锚点
回测（本地 Frangieh 真实 KO 方向对照 + mean-shift baseline 对照，判据预注册）。
LPM 脚本 `run_lpm_predictions.py` 保留：若未来获取 GWPS 65.83GB 数据（外部硬盘），
`trained_response` 证据源可按 §6a 命令重新启用（覆盖 APOE/MAPT/PSEN1 三候选）。

## 6c. AD 五候选分流（方案 §6，2026-09-18）

主环境用 `scripts/route_ad_candidates.py` 执行方案第 6 章的路线选择。它消费
轴审计、optional DEG/GRN/external evidence 与 APOE 锚点回测，写出
exploratory sidecar。2026-09-18 起默认决策为：observed 准入 =
signed-direction（FDR 0.05 只作报告标签）、KD 并入 KO、公共 Perturb-seq
out of scope。约束与产物见
[`ptm_activity_pipeline.md` §7.4](ptm_activity_pipeline.md)。

这条路径**不是** E2E 入口：`formal_invocation_allowed` 恒为 false；当前 fail
的 GEARS/Geneformer 资产保持 `lineage_boundary=supplementary_only`，即使未来
锚点 pass 也要等独立 formal consumer 获批。Geneformer `network_counterfactual`
只报告 `influence_score`，不能写入 `predicted_direction`。不得把 FDR>0.05
改标成显著，也不得从 KO 推导 KD。

## 7. 门禁状态历史快照（截至 2026-09-10）

> 下表保留 2026-09-10 的工程快照，表内旧测试日期不代表 2026-09-13 本次验证。
> 当前执行依据是 [`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md) 与
> [`DAVF/PerturbGen 中央方案`](../DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)；
> 本页前文的 2026-09-13 donor audit 仅按当前登记结果更新。

| Gate | 内容 | 状态 |
|---|---|---|
| Gate-0 M0⑤ | 独立环境 smoke（perturb 51.88s / 1757 MiB / h5ad schema 通过） | ✅ 已过（`outputs/perturbgen/spike/20260823_m0_smoke/evidence.json`） |
| Gate-0 M0⑥ | ≥3 donor 合规 cohort | ⚠️ between_donor 契约落地；GSE174367 AD 队列 preflight 7/7 PASS（数据契约验收，非生物学 PASS）；完整 E2E 队列运行仍待执行 |
| Gate-1~3 | 契约 / runner / 双路径统计 | ⚠️ 工程组件完成；E2E CLI 已接通方向 gate→runner，正式 donor/统计证据仍待补齐 |
| Gate-E | DAVF 新底座资产/接口回归 | ✅ current checkpoint 与真实 token/decoder 分离测试通过；生物学方向门仍待 held-out 验证 |
| Gate-4 | 真实 smoke / 正式 release evidence | ⚠️ 本地 Datlinger 750-cell 六阶段 smoke 已通过；正式 donor 队列仍待运行 |
| Gate-5 | 冻结队列科学验收（3 seeds / held-out / ≥99 null / BH-FDR） | ⚠️ 工具与独立重算入口已具备；真实 cohort 到位前未执行 |

## 8. 常见陷阱

- **不要把 scPerturb 通用数据当 donor cohort**（L-2026-0822-06）；
- **不要混装 scgpt 与 scvi-tools>=1.2 到同一环境**（TD-M05：scgpt 需要独立环境，且已从 `requirements-lock.txt` 移除）；
- **PerturbGen 不暴露 HTTP API**（方案 §9 有意决策：长任务不进同步 API）；
- 正式证据必须可提交可留存：benchmark JSON 写 `outputs/real_assets/`，不是 pytest 临时目录（L-2026-0822-05）。
