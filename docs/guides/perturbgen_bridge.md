# PerturbGen 桥接指南（DAVF × PerturbGen 双路径整合）

> **文档版本**：v1.7（2026-09-13，补充研究边界、公共准备生命周期与六阶段入口）
> **权威方案**：[`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](../DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)（v2.0）
> **详细执行方案**：[`docs/guides/davf_ko_kd_training.md`](davf_ko_kd_training.md)
> **状态基线**：[`project_analysis_20260914.md`](../../project_analysis_20260914.md)（本报告按源码、测试和真实资产边界逐项审计；工程契约与科学验收分开记账）

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

**AD 队列状态（2026-09-14，L-2026-0914-01）**：data/AD 四个 GEO 队列审计结论为
`GATE0_BLOCKED_SEMANTICS_AND_LABELS`（`outputs/perturbgen/spike/20260914_ad_cohort_audit/evidence.json`）；
Gate-0 新增 `between_donor` 配对后，GSE174367 已标准化为
`data/AD/standardized/GSE174367_ad_cohort.h5ad` 并以 between_donor preflight
7/7 细胞类型 PASS
（`outputs/perturbgen/spike/20260914_gse174367_gate0/evidence.json`）。
GSE157827/GSE188545/GSE147528 仍缺 donor/cell 注释或 cell calling，未合并。
该 preflight 是数据契约验收，不是生物学 PASS。

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
