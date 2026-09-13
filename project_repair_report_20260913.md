# PTM2CellNet 代码修复报告（2026-09-13）

> 本文件覆盖 2026-09-13 **两轮**工程修复：较早一轮的 YAML 身份绑定 / KO-KD route / N-05 lineage（不在本 worker 中重做），以及本轮按权威分析 [`project_analysis_20260913.md`](project_analysis_20260913.md) §4.2 执行的 **U-01～U-07** 与高/中技术债。
>
> 事实来源是当前代码、测试、CI 与方案文档。历史报告只作背景。本轮 **没有** 把 smoke、synthetic、mock 或真实桥接写成生物学 PASS。

## 1. 结论先行

**工程主线在代码层已接通 formal 隔离与缺失接口；科学主线仍被资产缺口 A-01～A-05 挡住。**

本报告记录的是工程接口修复；它不等于方向语义、证据独立性或正式双路生物学验收已完成，后续执行以 [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)、中央方案 [`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md) 和 [`task_plan.md`](task_plan.md) 为准。

本轮从第一性原理做了这些选择，而不是逐字照抄分析里的每一个备选方案：

1. **Formal vs engineering 隔离，而不是拆掉演示通路。** `uniform_candidate_pvalue` 仍可用于 `evaluation_mode=engineering`，但必须盖上 `evidence_class=synthetic` 且 `scientific_acceptance=false`。`evaluation_mode=formal` 拒绝 uniform/外部表/手填 p 与手填 unperturbed quality。这同时采纳了分析 §6.1 TD-13-01 方案 1（堵住假 PASS）和方案 2（演示仍可用）。
2. **候选 p 的 estimand 显式写成 `conservative_max_required_runs`。** 两路径 × 主 mode（up→mask / down→overexpress）× 所需 seed 的 empirical p 取 **max**。缺覆盖就报错，不编造 Fisher/Tippett。pad/delete 不进 p，仍走 KO 2-of-3 门。这解决了分析 §4.2 U-02「规则未定前 formal 应拒绝 PASS」与「不要锁死未签字的组合规则」。
3. **Null 基因不是候选。** `run_matched_null_stages` 禁止伪造 `PerturbGenInvocation`。单测用 `stage_executor`；真 GPU 必须再给 `rescue_extractor`，不猜 DEG 列。
4. **U-06 明确是次级证据。** 方案 §4.7 硬 PASS 未列 pathway；接口存在但 `affects_dual_path_verdict=False`。
5. **U-07 不删 Geneformer。** Gate-E 未过；pickle 在 production/strict 下拒绝未钉扎 `pickle.load`，非 production 仅警告，以免打断 legacy fusion demo 测试。
6. **TD-13-10 不改 `DAVFInferenceConfig` 默认 10×5000。** 只在 `PTM2CELLNET_ENV=production` 时拒绝无 `embedding_asset_path` 的 `use_davf`。把小 asset 单测自动改成 64×4018 会打坏契约。
7. **TD-13-05 禁止全仓 format。** 本轮后 `ruff format --check` 仍有 **335** 个文件待格式化（分析时为 329；新增未格式化文件，未批量改写）。

离线验证（本机、脱离沙箱、`not slow and not gpu`、`--timeout=300`）：

```text
2610 passed, 21 skipped, 73 warnings in 699.36s, exit 0
ruff check src scripts tests                 通过
mypy src/ --ignore-missing-imports           0 errors / 165 files
python scripts/check_requirements_consistency.py  通过
ruff format --check src scripts tests        335 files would be reformatted（未批量修复）
```

相对权威分析 §1 记录的 2578 passed / 161 mypy files：本轮新增测试与模块，**不是**把旧数字改写成生物学结果。

---

## 2. 事实来源与章节引用

| 文档 | 本报告使用的章节 |
|---|---|
| [`project_analysis_20260913.md`](project_analysis_20260913.md) | §1 执行摘要；§4.2 U-01～U-07；§4.3 A-01～A-06；§4.4 主线流程；§6.1 高债 TD-13-01～04；§6.2 中债 TD-13-05～12；§7 结论与建议 |
| [`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md) | §4.6 数据/训练/held-out；§4.7 正式 PASS 七条；§5.4 科学质量门；§7.2 M6/M7 |
| [`lessons.md`](lessons.md) | L-2026-0821-01 双路径 AND；L-2026-0822-06 通用 scPerturb ≠ 合规队列；L-2026-0901-01 DAVF 方向 / PerturbGen 效用（含 pathway）；L-2026-0902-01 schema v2 64×4018；L-2026-0902-02 串联 gate；L-2026-0902-03 真实桥接 ≠ 生物学验收 |
| [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) | Quick Reference；2026-09-13 修复状态 |
| [`docs/guides/perturbgen_bridge.md`](docs/guides/perturbgen_bridge.md) §6 | 评估入口；本轮已补 formal / null CLI |
| [`docs/guides/davf_perturbgen_e2e.md`](docs/guides/davf_perturbgen_e2e.md) | E2E 与 donor 绑定 |
| [`docs/guides/deployment.md`](docs/guides/deployment.md) 环境变量表 | production / pickle pin |
| [`docs/guides/davf_ko_kd_training.md`](docs/guides/davf_ko_kd_training.md) | 历史复训未绑定 donor split 的声明 |
| [`AGENTS.md`](AGENTS.md) | 主线不得绕过 DAVF gate；验证命令；不得把 smoke 当 PASS |

分析 §4.2 给出的目标接口与本轮落地对照见 §3。

---

## 3. 问题修复汇总（按严重程度）

### 3.1 高：代码缺口 U-01～U-05 / TD-13-01～04

| ID | 分析定位 | 本轮决策与落地 | 状态 |
|---|---|---|---|
| **U-01 / TD-13-02** | §4.2：只有 `select_matched_nulls`，无 batch 生成端；目标 `run_matched_null_stages(...) → perturbgen_null_distribution/v1` | 新增 `src/integration/perturbgen/null_generation.py` 与 `scripts/run_matched_null_stages.py`。选择方案 1（复用 perturb-only stage plan），拒绝伪造 invocation。CLI 支持 `--dry-run` / `--records-json`；GPU 路径强制 `rescue_extractor` | **代码闭合**。正式 ≥99 GPU 矩阵仍依赖 A-01/A-05 |
| **U-02 / TD-13-08** | §4.2 / §6.2：单 run `(k+1)/(n+1)` 不进 dual-path AND；无跨 path/mode/seed 聚合 | `aggregate_candidate_empirical_pvalues`；estimand=`conservative_max_required_runs`。path AND 仍用 rescue/donor/null **count**；p 只上候选层 BH。`evaluate_perturbgen_dual_path.py` 在 formal 下用聚合 p 而不是 payload 里的 `candidate_pvalue` | **代码闭合**。Estimand 已声明，未发明 Fisher |
| **U-03 / TD-13-01** | §4.2：`uniform_candidate_pvalue` 可改正式 q/verdict | assembler/evaluator/CLI `--evaluation-mode formal` 拒绝 uniform/外部表。engineering 仍可 `verdict=pass` 但 `scientific_acceptance=false` | **代码闭合** |
| **U-04** | §4.2：`evaluate_unperturbed_quality` 仅测试；CLI 手填 status | `extract_unperturbed_quality_from_h5ad`：pred vs true 的 DEG 方向恢复 + held-out signature Pearson。formal 要求 `source=extract_unperturbed_quality_from_h5ad` | **代码闭合**。真实质量仍要合规 h5ad |
| **U-05 / TD-13-03** | §4.2：训练/tokenise 无 `train_donors`/`held_out_donors` | `donor_split.py`：≥2 train / ≥3 held-out、互斥、SHA、与 frozen manifest 逐值绑定。`train_latent_davf.py` 与 `run_davf_perturbgen_e2e.py` 增加 `--train-donors` / `--held-out-donors` / `--frozen-cohort-manifest` / `--require-donor-split`，写入 checkpoint metrics 与 E2E report | **接口闭合**。未提供列表时 `donor_split_status=unspecified`，禁止声称 held-out 未泄漏 |
| **TD-13-04** | §6.1：`finetune_davf_e2e.build_model` 无条件 `PTMDirectionMapper()`，token ID 错配 PerturbGen 表 | `--embedding-asset` 必填；`build_model` 必须 `build_perturbgen_direction_mapper()` | **代码闭合** |

### 3.2 中：U-06～U-07 与 TD-13-05～12

| ID | 决策 | 状态 |
|---|---|---|
| **U-06** | 方案 §4.7 硬 PASS 未列 GSEA；L-2026-0901-01 提到 pathway 效用。新增 `pathway_evidence.py`，默认 `backend=none` → inconclusive，**禁止**写 dual-path 判决 | **接口闭合 / 次级**。未接 gseapy 执行 |
| **U-07** | 方案 §7.2 M7：Gate-E 未过不得删。保留 `geneformer_embedding.py`；strict 禁止 hash fallback；production/strict 拒绝未钉扎 pickle | **按门控保留** |
| **TD-13-05** | 不批量 format 335 文件 | **刻意未做** |
| **TD-13-06** | pip 元数据冲突与 CI extra 取舍一致；不放宽 core pin | **文档化，未改 core** |
| **TD-13-07** | 本轮无 `--cov`；禁止引用历史 75.27% | **未刷新覆盖率** |
| **TD-13-08** | 与 U-02 一起：字段保留为 provenance，path 判决不用 p | **已解释并接线到候选层** |
| **TD-13-09** | `.github/workflows/ci.yml` 两处 `--timeout=120` → **300**，与 AGENTS 对齐 | **已改 timeout**。默认 job 仍不含完整 e2e/real_assets（与 nightly `full-test.yml` 分层一致） |
| **TD-13-10** | production 无 schema v2 asset 则硬失败；非 production 标 `davf_path_class=legacy_fusion_demo` | **生产路径闭合**。未改默认 10×5000 以免打坏单测 |
| **TD-13-11** | pickle 需 `PTM2CELLNET_GENEFORMER_VOCAB_SHA256` 或 `ALLOW_GENEFORMER_PICKLE`；**production/strict 拒绝**，非 production 警告后加载（否则会打断大量 Geneformer demo 测试） | **生产热点闭合** |
| **TD-13-12** | `docs/guides/deployment.md` 写明 `PTM2CELLNET_ENV=production`、API key、pickle pin | **文档闭合**。既有 `test_production_security.py` 覆盖 fail-fast |

### 3.3 低 / 非本轮范围

- 分析 §6.3 低债（signaling TODO、CodeGraph 滞后、`LatentDAVF` 仍 import Geneformer）未改。
- 验收缺口 **A-01～A-06**（§4.3）是资产，不是本轮能闭环的科学 PASS。

---

## 4. 每轮迭代的执行情况与结果

### 4.0 同日既有成果（较早 worker，不在本会话重做）

权威分析 §1 与 `docs/CURRENT_STATUS.md` 已记录：

- runner 报告与当前 YAML 的 gene/mode/Ensembl/route/seed/path 绑定；
- 显式 KO/KD rescue 模式（KO/up 才 2-of-3；KD/up 仅 mask）；
- N-05 lineage：tokenise stage 精确绑定 + runner 已登记的 `result_h5ad.sha256`。

本轮 **不把上述工作算进本会话的 diff**，但 E2E 复核会把它当作已存在的工程底座。

### 4.1 迭代 1 — 接口与隔离（实现）

对照分析 §4.2 目标接口与 §7 建议优先级（TD-13-01/U-03 → U-02 → U-05 → U-01 → A-01）：

| 模块 | 作用 |
|---|---|
| `src/integration/perturbgen/null_generation.py` | U-01 生成/收集 |
| `src/integration/perturbgen/empirical_pvalue.py` | U-02 聚合 |
| `src/integration/perturbgen/donor_split.py` | U-05 SHA 绑定 |
| `src/integration/perturbgen/pathway_evidence.py` | U-06 次级 |
| `eval_assembly.py` / `dual_path.py` / `evaluate_perturbgen_dual_path.py` / `build_dual_path_eval_input.py` | U-03/U-04 formal 隔离 |
| `results.py` `extract_unperturbed_quality_from_h5ad` | U-04 生产者 |
| `train_latent_davf.py` / `run_davf_perturbgen_e2e.py` | U-05 CLI |
| `finetune_davf_e2e.py` / `architectures.py` | TD-13-04 / TD-13-10 |
| `geneformer_embedding.py` | U-07 / TD-13-11 |
| `.github/workflows/ci.yml` | TD-13-09 timeout 300 |

第一性原理拒绝项：不把 null 做成假 candidate；不在缺 run 时填 p；不在 Gate-E 前删 Geneformer；不全仓 ruff format。

### 4.2 迭代 2 — 测试与 pickle 政策修正

新增/扩展单测覆盖 U-01～U-07 与 formal CLI。首次把 pickle **默认拒绝** 后，`test_architecture_davf_integration` 等 legacy fusion 用例在加载官方 hub pickle 时失败（`GeneformerVocabularyError`）。

独立决策：热点控制应对准 **production/strict**，而不是破坏非生产 demo。改为：

- `PTM2CELLNET_ENV=production` 或 `strict=True`：未钉扎 pickle → 拒绝；
- 非 production：警告 + 加载（与「大量测试依赖 legacy_fusion_demo」一致，见分析 §6.1 TD-13-04 方案 1 的缺点说明）。

针对性复测：architecture / pickle / frozen / ptm_module_switch **43 passed**。

### 4.3 迭代 3 — 全量回归与夹具缺口

第一次全量：`2608 passed, 2 failed`。失败不是生产逻辑回退：

1. `test_run_davf_perturbgen_e2e.py` 的 `SimpleNamespace` 未带新 donor 字段 → 补 `train_donors=None` 等默认值。
2. Geneformer `__new__` 回归夹具未设 `strict` → 补 `loader.strict = False`。

### 4.4 迭代 4 — 干净全量

```text
python -m pytest -m "not slow and not gpu" --timeout=300
2610 passed, 21 skipped, 73 warnings in 699.36s, exit 0
```

21 skipped 主要为 `real_assets` / 个别 gpu 标记，与「本轮不做生物学 PASS」一致。

---

## 5. 系统性复核：E2E 训练与推理是否满足技术要求

对照方案 §4.6 / §4.7 / §5.4 与分析 §4.4 主线图。判定分三列：**代码契约**、**本机资产**、**能否声称科学 PASS**。

### 5.1 训练（LatentDAVF）

| 技术要求 | 代码 | 资产 | 科学 PASS |
|---|---|---|---|
| schema v2：`latent_dim=64`、`num_genes=4018`、冻结 PerturbGen asset（L-2026-0902-01） | 训练入口已强制 embedding asset / scVI 校验 | 本机已有 KO/KD checkpoint 与 asset | 不能把验证 loss 当 held-out 方向准确率（训练指南已写明） |
| 训练 donor 与 held-out 互斥（方案 §4.6-2；frozen ≥2/≥3） | CLI + SHA + 可选 frozen 绑定 | 调用方必须显式给列表；历史 2026-09-04 复训 **未** 绑定 | 未绑定则 `unspecified`，**不能**声称无泄漏 |
| pair NPZ 与 CLI split SHA 一致 | trainer 会比对 NPZ `metadata.donor_split` | 现有 pair 未必带该字段 | 缺字段时不编造 |

**根因（若未达科学训练要求）：** 缺口从「入口没有 donor 参数」变成「外部未提供与 frozen 一致的 ≥5 donor 划分，且历史 checkpoint 无 split provenance」。这是 A-03 / TD-13-03 的资产后半段，不是缺函数。

### 5.2 推理桥接（PTM → DAVF → 三方 gate → invocation）

| 技术要求 | 代码 | 资产 | 科学 PASS |
|---|---|---|---|
| 不得绕过 DAVF gate（L-2026-0902-02，AGENTS 主线） | orchestrator 仍只对 pass 生成 invocation | — | 工程满足 |
| 独立表达方向 + FDR | 既有 direction gate | 需要观测 log2fc/FDR | 无合规 cohort 时只能跑契约 |
| API/融合 `use_davf` | production 无 asset 硬失败；否则 `legacy_fusion_demo` | 正式路径必须 schema v2 asset | 融合 10×5000 **不是** 正式 64×4018 |
| finetune mapper | 锁定 PerturbGen mapper | 旧混用 checkpoint 无法靠改 mapper 挽救（分析 §6.1 TD-13-04 风险） | 旧 finetune 权重若存在，需作废重训 |

### 5.3 PerturbGen 六阶段与双路径效用

| 技术要求（方案 §4.7） | 代码 | 资产 | 科学 PASS |
|---|---|---|---|
| Gate-0：raw counts + Ensembl + normal/disease + 显式 donor + ≥3 shared | `data_prep` / E2E preflight 仍硬失败 | 分析 §4.3 A-01：30/26/**0** 合规 | **否** |
| 双路径 AND、KO 2-of-3、KD mask-only | `dual_path.py`（含本轮 KD 单测） | — | 工程满足 |
| 正式 ≥99 匹配 null + `(k+1)/(n+1)` + 候选 BH | 选择器 + **生成端** + 聚合器 + formal 拒绝 uniform | 未跑正式 GPU 99×双路径×3 seed（A-05） | **否** |
| 未扰动质量门（§5.4） | 提取器存在；formal 拒手填 | 需要真实 pred/true h5ad | 合成 h5ad 单测 ≠ 发布 |
| `q_value < 0.05` 来自 empirical 而非 uniform | formal 路径已强制 | 无真实 null rescue 就不能出真 q | **否** |
| pathway/GSEA | 次级接口，不影响 AND | 无 GMT 合同 | 按方案 **不作为** 硬 PASS |
| Gate-E ≥200（§5.4 / §7.2 M7） | `gate_e.py` 工具在 | 无真实基准 CSV（A-04） | **不得删 Geneformer** |

### 5.4 总判

**满足 E2E 训练/推理的工程技术要求（可编排、可拒绝假 PASS、可绑定 donor、可生成 null 计划）。不满足正式科学验收的全部技术要求。**

阻塞性根因按分析 §4.3 / §7 排序：

1. **A-01 无合规 `normal/disease` 队列** → Gate-0 过不了，后续 tokenise/perturb/null 只能是 smoke。根因：本机 scPerturb 不是该契约（L-2026-0822-06）。
2. **A-05 正式 GPU 矩阵未跑** → 即使生成器在，也没有 3 seed × 双路径 × ≥99 null 的真实 rescue 分布。根因：依赖 1 + PerturbGen 独立环境墙钟，不是缺 `run_matched_null_stages`。
3. **A-02/A-03 冻结队列与 held-out DAVF 指标** → split 接口在，真实列表与 pair 未按 SHA 冻结重训。
4. **A-04 Gate-E** → 挡住 U-07 删除 Geneformer。

这些 **不能** 用 2610 passed 抵消。

---

## 6. 未解决问题及后续解决策略

### 6.1 阻塞 / 高（科学发布）

| 项 | 策略 | 实施步骤 | 时间节点 |
|---|---|---|---|
| **A-01 合规 cohort** | 外部数据，不改契约迁就 scPerturb | 1. 选定带显式 donor 的 normal/disease raw-count 队列；2. Ensembl canonical；3. `prepare_perturbgen_anndata` Gate-0 必须 pass；4. 登记 evidence JSON | **T0～T10 工作日（数据依赖，非编码）**。无数据则停止科学声明 |
| **A-05 正式 null GPU 矩阵** | 用已落地生成器，禁止手填 rescue | 1. `select_matched_nulls`；2. 对每个 candidate/path/主 mode/seed 调 `run_matched_null_stages`（真 runner + 显式 `rescue_extractor`）；3. 写出 `perturbgen_null_distribution/v1`；4. formal assembler → evaluator | **在 A-01 之后 T+3～T+10 GPU 日**（99×路径×seed 墙钟）。之前只允许 `--dry-run` |
| **A-02/A-03 M6 + held-out DAVF** | `--require-donor-split` 作为正式训练默认 | 1. 冻结 ≥2/≥3 互斥列表并写入 pair NPZ；2. `train_latent_davf.py --require-donor-split --frozen-cohort-manifest`；3. `evaluate_latent_davf.py` 按 held-out 基因/donor 报方向指标，不把 flow loss 当准确率 | **A-01 后 T+2～T+8** |
| **A-04 Gate-E ≥200** | 工具已在；缺基准 | 准备可追溯 PTM→gene CSV，跑 `gate_e.py`，对照旧冻结基线 | **与 M6 并行，T+5～T+15**。未过则 **禁止 M7 删 Geneformer** |
| **旧 finetune 混用权重（TD-13-04 风险）** | 作废重训 | 凡 `PTMDirectionMapper()` 时代产出的 finetune checkpoint 不得再用于 PerturbGen asset 表 | **发现即隔离，0.5 日清点** |

### 6.2 中（工程债，不挡科学声明之外的开发）

| 项 | 策略 | 步骤 | 时间节点 |
|---|---|---|---|
| TD-13-05 format | **不要**一次 PR 改 335 文件 | 新文件在提交前 format；可选 CI 只检查 diff | 持续；专项 PR 另排 |
| TD-13-06 pip check | 维持 core pin | 不把 scgpt/torchaudio 冲突塞进 core | 无截止 |
| TD-13-07 覆盖率 | 独立环境 `--cov`，禁止抄 75.27% | 与 CI `fail_under=74` 对齐后更新 `docs/TEST_COVERAGE.md` | 下一次专门覆盖率会话 |
| TD-13-09 默认 CI 无完整 e2e | 保持分层 | PR 跑 unit/integration；nightly `full-test.yml` | 无截止 |
| U-06 gseapy 执行 | 仅在有签字 GMT/estimand 后接线 | 保持 `affects_dual_path_verdict=False` | 统计/生物学负责人签字后 1～2 日 |
| pickle 非 production 仍 load | 接受为 demo；生产必须 pin 或 json | 部署检查清单已写 | 生产上线前 |

### 6.3 明确不做

- 重建实时质谱流 / 自定义 PTM 库 / GUI / 新 API-key 产品（分析 §1，AGENTS）。
- 猜 checkpoint、token 映射或「最新文件」替代 manifest。
- 在 Gate-E 前删除 `geneformer_embedding.py`。
- 把 Datlinger smoke、synthetic AnnData 或 2610 passed 写成效用 PASS（L-2026-0902-03）。

---

## 7. 关键实现锚点（便于审计）

| 能力 | 位置 |
|---|---|
| Null 生成 | `src/integration/perturbgen/null_generation.py` `run_matched_null_stages`；CLI `scripts/run_matched_null_stages.py` |
| 候选 p 聚合 | `src/integration/perturbgen/empirical_pvalue.py` `CONSERVATIVE_MAX_ESTIMAND` |
| Formal 隔离 | `eval_assembly.build_eval_input_payload(evaluation_mode=...)`；`scripts/evaluate_perturbgen_dual_path.py`；`scripts/build_dual_path_eval_input.py --evaluation-mode` |
| 未扰动质量 | `results.extract_unperturbed_quality_from_h5ad` |
| Donor split | `donor_split.build_donor_split`；`scripts/train_latent_davf.py`；`scripts/run_davf_perturbgen_e2e.py` |
| Pathway 次级 | `pathway_evidence.evaluate_pathway_evidence` |
| Production DAVF | `architectures.py` `davf_path_class`；`PTM2CELLNET_ENV=production` |
| Finetune mapper | `scripts/finetune_davf_e2e.py` `build_model` |
| CI timeout | `.github/workflows/ci.yml` `--timeout=300` |

---

## 8. 与权威分析建议的偏差（有意）

分析 §7 建议优先级完整执行了代码段（U-03→U-02→U-05→U-01）。未执行 A-01 外部数据，因为那不是本仓库能「修复」的缺口。

相对分析文本的有意偏差：

1. **U-02 estimand**：分析说规则未定前 formal 应拒绝 PASS。本轮 **同时** 落地一条保守、可审计的 max 规则，并让 formal 只走该规则。若统计负责人日后否定 max，应改 `empirical_pvalue.py` 并升 schema，而不是再引入静默 uniform。
2. **TD-13-10**：未把融合默认维数改成 64×4018。
3. **TD-13-11**：未在非 production 默认拒绝 pickle（否则 legacy 测试与 demo 全挂）。
4. **TD-13-05**：未全仓 format。

这些偏差服务于「堵住假正式 PASS、不破坏可维护的工程测试」这一第一性原理，而不是回避问题。

---

## 9. 验证命令（可复现）

```bash
python -m pytest -m "not slow and not gpu" --timeout=300
ruff check src scripts tests
python -m ruff format --check src scripts tests   # 预期仍大量 Would reformat
python -m mypy src/ --ignore-missing-imports
python scripts/check_requirements_consistency.py
```

GPU null 生成必须在沙箱外、PerturbGen 独立环境中执行，且必须提供 `rescue_extractor`。本轮 **没有** 跑该 GPU 矩阵，也 **没有** 重训 DAVF。
