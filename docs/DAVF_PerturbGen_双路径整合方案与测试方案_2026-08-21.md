# DAVF × PerturbGen 双路径整合方案与测试方案（v2.0 重新评估版）

> **文档编号**：PROPOSAL-20260821-01  
> **版本**：v2.0  
> **重新评估时间**：2026-08-21～2026-08-22  
> **状态**：已批准实施；M1–M3 工程主干已完成，M0/Gate-0 因真实资产缺失保持阻断（2026-08-22）  
> **决策依据**：`lessons.md` L-2026-0821-01、L-2026-0821-02  
> **代码基线**：当前工作区 + PerturbGen `ref/Perturbgen-src` commit `a9a9375`  
> **替代版本**：本版完整替代 v1.1；保留“双路径严格 AND”和“移除 Geneformer”两项用户决策，但修正其技术实现

---

## 1. 审核结论

| 评估项 | 结论 | 说明 |
|---|---|---|
| PerturbGen `src/tgt` 双路径 | **有条件可行** | 官方代码直接支持 `perturbation_sequence=["src"]/["tgt"]`、`pert_tps` 和 `mask/pad/delete/overexpress` |
| 与现有 PTM2CellNet 集成 | **兼容性较高** | 适合沿用 frozen dataclass、lazy import、配置集中、报告 artifact 等现有模式；PerturbGen 放独立环境，不改 API 契约 |
| DAVF 底座切换 | **可行性中等，必须先做真实权重试验** | 官方没有“导出静态 token embedding safetensors”的稳定命令；必须先验证 checkpoint 键、词表行号和实际 embedding 层 |
| 科学判定 | **当前方案方向正确，但 v1.1 门槛不足** | 需要 held-out donor、至少 3 个可评估 donor、3 seeds、匹配 null、BH-FDR 和 `PASS/FAIL/INCONCLUSIVE` 三态 |
| 实施复杂度 | **高** | 不是单纯新增适配层，还会修改 DAVF runtime、配置加载、CI、真实资产测试和模型资产协议 |
| 预期效果 | **工程收益确定，模型收益待证** | 可确定消除未知基因哈希映射、统一词表和形成可复现双路径流水线；不能预先承诺 DAVF 指标或候选质量提升 |
| 是否建议按 v1.1 直接实施 | **否** | 必须先完成 M0 可证伪试验，再按本版阶段门执行 |

最终候选仍执行项目既定的严格标准：只有早期干预路径和治疗干预路径均达到正式门槛才可 `PASS`。任一路径因无表达、无 token、资产失败或 donor 不足而不可评估时，结论是 `INCONCLUSIVE`，不是生物学 `FAIL`，更不能进入候选清单。

---

## 2. 当前代码现状

### 2.1 仓库规模与结构

本轮实查：CodeGraph 已索引 367 个文件、7,847 个符号节点和 8,986 条边；文件系统中有 134 个 `src/**/*.py`、175 个 `tests/**/*.py`、48 个 `scripts/**/*.py`、30 个配置 YAML。测试目录包含 168 个 `test_*.py`。

| 层 | 当前实现 | 与本方案关系 |
|---|---|---|
| 数据 | `src/data/`：AnnData/表格契约、预处理、特征、同源划分、DAVF 增强 | 可复用数据契约思想，但 PerturbGen 需要单独的 raw counts/ENSG/tokenisation 契约 |
| 模型 | `PTM2CellNetBase`、多类序列编码器、DAVF/LatentDAVF、PTM mapper | DAVF 底座迁移会触及 runtime 主链 |
| 训练 | 自研 Trainer + Lightning，安全 checkpoint、artifact manifest、TensorBoard | 可复用 artifact/provenance；不能复用 PerturbGen 训练环境 |
| 评估 | 分类/回归/排序/置信区间、独立 `evaluate/cross_validate` | rescue/null/donor 统计应放新集成包，不污染通用指标 |
| 集成 | GenKI、PTM virtual perturbation、frozen contracts、lazy import | 是 PerturbGen 主进程适配层的直接工程模板 |
| API | FastAPI、限流、请求体限制、API key 兼容、Prometheus/GPU 指标 | 本期不暴露 PerturbGen API，避免长任务进入同步请求链 |
| 测试 | unit/integration/e2e/real_assets 分层，主 CI + nightly | 单元底座成熟，但现有 workflow 都跳过 `slow/gpu`，必须新增真实资产门禁 |

### 2.2 技术栈现状

| 项 | 主项目 | PerturbGen 参考实现 | 判断 |
|---|---|---|---|
| Python | 仓库/CI 基线 `>=3.10`；当前交互 shell 为 3.12.13，`pytest` 可执行文件使用系统 Python 3.10.12 | `>=3.11` | 解释器来源已存在差异，更应使用显式独立环境 |
| PyTorch | lock 为 `2.4.1+cu118` | 固定 `2.5.1` | 不合并环境 |
| Lightning | lock 为 `2.6.5` | 固定 `2.4.0` | 不合并环境 |
| datasets | lock 为 `2.21.0` | 固定 `3.0.0` | 不合并环境 |
| anndata | analysis 为 `<0.12`，lock `0.11.4` | `0.11` | 版本表面兼容，但整套依赖仍应隔离 |
| 其他 | FastAPI/Pydantic、scVI、PyG、Transformers、Mamba、safetensors | 大量固定科学计算/CUDA依赖 | 子进程边界最稳健 |

主项目依赖已拆为 core/pretrained/mamba/analysis/dev/docs。默认 CI 只安装 core + dev；nightly 聚合依赖也不等于 PerturbGen 环境，因此不能把“主 CI 通过”当成外部链路通过。

### 2.3 DAVF 当前真实调用链

```text
API/训练 batch
  -> PTM2CellNetBase (src/models/architectures.py)
     -> PTMDirectionMapper
        -> gene_ids + direction code(KO/KD/OE) + attention_mask
     -> DAVFInferenceModule
        -> LatentDAVF 或 GeneMLEPEncoder
        -> DeltaProjection
     -> 与主模型特征融合
```

必须纠正 v1.1 的三个判断：

1. `PTMDirectionMapper` 的 `direction` 是由 PTM 类型/上下文映射出的**干预动作代码**，不是 normal/disease 实测表达的 `up/down`。
2. `DAVFInferenceModule` 当前实例化 `LatentDAVF(latent_config)` 时没有传入 `GeneformerEmbeddingLoader` 或预训练 embedding；`geneformer_path` 只存在于配置/警告，没有接到 runtime embedding。
3. 当前 Geneformer loader 主要为 mapper 提供 `_gene_to_idx`；未知 gene symbol/ENSG 会经 SHA-256 哈希落到某一行。它会产生语义错误和碰撞风险，但“替换 loader 必然改变 DAVF 输入维度”并不成立。只有真正把 PerturbGen embedding 注入 `LatentDAVF` 后，才涉及投影、参数形状和重训。

现有 `LatentDAVF` 已支持 `pretrained_gene_embeddings` 和维度投影，这是底座迁移可行的主要基础；真正的 runtime 改点是 `architectures.py + davf_inference.py + latent_davf.py + ptm_direction_mapper.py + configs/davf_integration.yaml`，而不是只改 `davf.py`。

### 2.4 PerturbGen 当前真实契约

| 能力 | 已验证事实 | 方案要求 |
|---|---|---|
| CLI | `tokenise/train-mask/train-decoder/extract-embedding` | 扰动另走 `perturbgen/Perturb/val.py --config` |
| 扰动模式 | `mask/pad/delete/overexpress` | KO 模式只保留目标 token 实际存在的细胞；OE 不过滤 |
| 双路径 | `src/tgt` + `pert_tps` | normal→disease 用 src；disease within-state 用 tgt |
| 输入 | raw counts、`var[ensembl_id]`、`obs[n_counts]`、状态/细胞类型/donor | 主进程逐项校验，不从 log `X` 猜 counts |
| 训练资产 | encoder → masking checkpoint → count decoder checkpoint | 三者路径、哈希、配置必须分别记录 |
| 输出 | h5ad：`X=pert_counts`、`layers[true_counts/pred_counts]`、embedding/cosine 元数据 | parser 按字段和 shape 严格校验 |
| 网络 | `PerturberTrainer.__init__` 在 inference 也会 `evaluate.load("rouge")` | 真环境镜像必须预热缓存或明确允许网络；不能只把风险归到 generate |
| embedding 导出 | 官方 `extract-embedding` 产出样本级 embedding h5ad | 静态 gene token matrix 导出是**项目自定义能力**，必须先验证真实 checkpoint |

---

## 3. 对 v1.1 的必要修订

| v1.1 问题 | v2.0 修订 |
|---|---|
| 把 mapper action 当 disease expression direction | 新增 `CandidateEvidence`；扰动方向以 held-out-safe normal/disease 差异表达方向为准，DAVF 证据单独记录 |
| 宣称全部为新增文件 | 明确新增适配层，同时修改 DAVF runtime、配置、CI 和文档 |
| 只列 `davf.py` 等注入点 | 增加真实主链 `architectures.py`，并区分直接 DAVF 类与 runtime wrapper |
| `export-embeddings` 写成官方能力 | 改为 M0 自定义可证伪试验；真实 checkpoint 不通过即阻断底座迁移 |
| stage 数一处 4、一处 5 | 固定六个可记录 stage：`tokenise/train_mask/train_decoder/perturb/export_gene_embeddings/report`；其中 export 属底座迁移支线 |
| 脚本路径前后不一致 | CLI 固定为顶层 `scripts/run_perturbgen_pipeline.py`；包内只放库代码 |
| `resume=文件存在` | 改为 manifest 指纹匹配：数据、代码、配置、权重、词典、stage 版本全部一致才复用 |
| 非 strict 时随机 fallback | 新链路禁止随机或零向量 fallback；资产缺失/词表未命中直接报错或标 `INCONCLUSIVE` |
| donor 阈值 `>0.5/≥0.5` 混用 | 统一严格多数 `>0.5`，且正式判定要求 `evaluable_donors >= 3` |
| 20 个 null 就做 `p<0.05` | 20 个只用于 smoke；正式验证至少 99 个匹配 null，并做加一经验 p 值和 BH-FDR |
| tiny smoke 同时承担科学验收 | 分成工程 smoke 与冻结真实队列科学验收，二者不得混称 |
| 10 gene/hour@A100 写成硬承诺 | 改为固定硬件/数据下建立基线，后续 P50/P95 与峰值显存回归不超过阈值 |

---

## 4. 修订后的目标架构

### 4.1 两条工作流

```text
工作流 A：PerturbGen 双路径（主要交付）
DAVF/队列候选 -> CandidateEvidence -> 数据/表达/token preflight
 -> PerturbGen 独立环境训练资产
 -> source intervention + disease within-state
 -> donor/seed/null/mode 统计
 -> PASS | FAIL | INCONCLUSIVE

工作流 B：DAVF 底座迁移（同一方案范围，独立门控）
真实 PerturbGen encoder ckpt
 -> 自定义静态 gene embedding 导出试验
 -> safetensors + vocabulary.json + manifest.json
 -> GeneVocabularyResolver + PerturbGenEmbeddingLoader
 -> DAVF runtime 注入 + 重训
 -> Gate-E
 -> 删除 Geneformer 主链与专属资产
```

工作流 A 不应被工作流 B 的研究不确定性阻塞。两者在 M5 前汇合：正式候选报告必须注明 DAVF evidence provenance；若 B 尚未通过，则只能产出工程/探索报告，不能宣称“统一底座已完成”。

### 4.2 主进程与外部环境边界

| 主项目环境负责 | PerturbGen 独立环境负责 |
|---|---|
| 输入契约、路径安全、symbol↔ENSG 解析 | 官方 tokenisation |
| 配置生成、manifest、stage 调度 | masking/count decoder 训练 |
| 输出 schema 校验、rescue/null/donor 统计 | `src/tgt` 扰动推理 |
| 报告、provenance、状态机 | 自定义 gene embedding 导出脚本 |

主进程绝不 import `perturbgen`。所有命令使用参数数组调用，禁止 `shell=True`；cwd 固定到校验后的 repo root；日志去除绝对敏感路径；stage 必须有 timeout、退出码和输出 schema 校验。

### 4.3 候选与方向契约

```python
@dataclass(frozen=True)
class CandidateEvidence:
    gene_symbol: str
    ensembl_id: str
    cell_type: str
    ptm_context: str
    observed_log2fc: float
    observed_fdr: float
    observed_direction: Literal["up", "down"]
    davf_action: Literal["ko", "kd", "oe"] | None
    davf_score: float | None
    davf_provenance: str

@dataclass(frozen=True)
class PathResult:
    status: Literal["evaluable", "inconclusive", "failed"]
    path: Literal["source_intervention", "within_state"]
    mode: Literal["mask", "pad", "delete", "overexpress"]
    rescue_excl_target: float | None
    evaluable_donors: int
    donor_consistency: float | None
    seed: int
    output_h5ad: str
    reason_code: str | None

@dataclass(frozen=True)
class DualPathVerdict:
    verdict: Literal["pass", "fail", "inconclusive"]
    q_value: float | None
    reasons: tuple[str, ...]
```

方向规则：

| 实测 disease vs normal | PerturbGen 主干预 | DAVF 字段的作用 |
|---|---|---|
| `up` 且 FDR 达标 | KO/KD：`mask` 主模式，`pad/delete` 敏感性 | 记录是否支持该干预，不替代实测方向 |
| `down` 且 FDR 达标 | OE：`overexpress` | 同上 |
| 不显著或方向不稳定 | `INCONCLUSIVE` | 不强行生成方向 |

若业务坚持由 DAVF 直接给出表达方向，必须新增并验证“DAVF gene-level delta 输出”契约；现有 `PTMDirectionMapper` 不满足该语义，禁止直接转换。

### 4.4 新增文件

| 文件 | 动作 | 验证 | 完成标准 |
|---|---|---|---|
| `src/integration/perturbgen/contracts.py` | 定义环境、队列、候选、stage、结果和 verdict | 纯单元测试 | 非法状态/路径/方向全部 fail-fast |
| `env_guard.py` | Python/repo/依赖/权重/词典/缓存探测 | fake env + 真环境 smoke | 一次返回完整问题清单，执行入口仍 fail-fast |
| `data_prep.py` | raw counts、ENSG、obs、覆盖、表达筛查 | 合成 AnnData | 无隐式 counts 转换；全量 reason code |
| `config_builder.py` | 官方 CLI 参数和 perturb YAML | golden schema | 与锁定 commit 字段一致 |
| `runner.py` | 六类 stage、GPU 串行锁、timeout、manifest | subprocess mock | 非零码/超时/脏输出/指纹不匹配均不可复用 |
| `results.py` | h5ad schema、signature、rescue、null、FDR | 已知答案 fixture | 排目标基因假阳性回归通过 |
| `dual_path.py` | 两路径、多 seed、多 mode 编排与三态判定 | mocked integration | 严格 AND，不混淆失败与不可评估 |
| `reports.py` | CSV/JSON/Markdown/evidence manifest | schema snapshot | 结果可追溯且字段稳定 |
| `embedding_export.py` | 外部导出命令与产物校验 | M0 真权重试验 | row/token/维度/hash 一致 |
| `src/models/gene_vocabulary.py` | 公共 ENSG/symbol→token resolver | collision/coverage tests | 不允许 hash fallback |
| `src/models/perturbgen_embedding.py` | 只读 safetensors + manifest | asset corruption tests | 资产不匹配直接失败 |
| `scripts/run_perturbgen_pipeline.py` | 离线 CLI | CLI integration | stage 选择、dry-run、resume 明确 |
| `scripts/export_perturbgen_gene_embeddings.py` | 仅由独立环境执行的自定义导出器 | real-assets | 不依赖主环境 import |
| `scripts/benchmark_perturbgen.py` | 固定 workload 性能/显存/磁盘报告 | benchmark artifact | 可重复比较 |
| `configs/integration/perturbgen.yaml` | 无机器绝对路径的模板 | config validation | 路径由环境变量/CLI 提供 |

### 4.5 修改文件与兼容策略

| 文件 | 修改 | 影响范围 | 兼容要求 |
|---|---|---|---|
| `src/models/architectures.py` | 注入共享 resolver/embedding asset config | API、训练、模型构造 | 默认不开启新链路时行为不变；正式启用时资产缺失 fail-fast |
| `src/models/davf_inference.py` | 删除死的 `geneformer_path` 语义；向 `LatentDAVF` 传预训练矩阵/词表 | DAVF runtime/checkpoint | checkpoint schema 显式版本化，不做 silent partial load |
| `src/models/latent_davf.py` | 接受共享 resolver 与导出矩阵，固定 row/token 对齐 | DAVF 训练/推理 | 已有 constructor 兼容窗口只用于迁移期测试 |
| `src/models/ptm_direction_mapper.py` | 用公共 resolver，变量从 `geneformer_loader` 改为语义化接口 | 所有 DAVF 调用 | direction code 业务逻辑保持不变 |
| `src/models/davf.py` | 类型改为 Protocol/公共 resolver | 直接 DAVF 类和测试 | 不作为唯一 runtime 注入点 |
| `configs/davf_integration.yaml` | 新资产 manifest/strict 开关 | 配置加载 | 不提供随机 fallback 配置 |
| `scripts/finetune_davf.py` 及 E2E 脚本 | 新词表/embedding/manifest 参数 | 重训 | 导出训练 provenance |
| `src/integration/__init__.py` | lazy export 新包 | import 行为 | core 环境 import 不触发 anndata/PerturbGen |
| `requirements-analysis.txt` / `setup.py` | 只声明主进程已有轻依赖 | 依赖契约 | 不把 PerturbGen 固定依赖放入主项目 |
| `.github/workflows/` | 增加 analysis 依赖 PR job 与 opt-in real-assets/GPU workflow | CI 成本 | core job 不增重；PerturbGen 主进程单测在 analysis job 必跑 |
| `pytest.ini` | 注册 `benchmark/real_assets/gpu` 标记 | 测试收集 | 避免未注册标记和误跑 |

迁移期必须同步修改 `tests/unit/test_architecture_davf_integration.py`、`tests/unit/test_davf_inference.py`、`tests/unit/models/test_geneformer_embedding.py`、`tests/integration/test_davf_pipeline.py` 及所有读取 `geneformer_path` 的配置断言。发布版采用配置 schema v2，一次性移除 `geneformer_path`，提供显式迁移检查/报错，不在 runtime 保留别名或静默兼容；Gate-E 的旧基线由冻结旧版本独立运行。

### 4.6 数据、训练与恢复语义

1. 数据必须含 raw counts、无版本 ENSG、唯一 obs_names、cell type/state/donor；目标 cell type 每状态至少 3 个可评估 donor。
2. normal/disease signature 只能用训练 donor 建立；rescue 判定在 held-out donor 上完成，避免数据泄漏。
3. `n_counts` 可从 raw counts 行和幂等生成并记录；其他关键字段不自动猜测。
4. checkpoint 复用由 `stage_manifest.json` 决定。指纹至少包含输入文件 SHA-256、数据 schema 摘要、PerturbGen commit、Python/依赖、完整配置、随机种子、权重/词典哈希、导出器版本。
5. 指纹不一致时拒绝 resume；不得覆盖旧输出，生成新 run_id。
6. GPU stage 默认串行。并行只允许显式配置并通过峰值显存预算，避免多个子进程争用同一 GPU。
7. 运行前估算输出磁盘占用；可用空间低于“抽样估算的 2 倍 + 10 GiB”时拒绝启动。默认不自动删除原始 h5ad。

### 4.7 rescue 与正式判定

疾病分数：

```text
S(X) = mean(Z_g, g in D_up) - mean(Z_g, g in D_down)
R = S(unperturbed_prediction) - S(perturbed_prediction)
```

要求：

- `D_up/D_down` 在训练 donor 上定义；每个候选构建判定分数时排除目标基因。
- 每个 held-out donor 单独计算 R；只在可评估 donor 上统计一致率。
- 至少 3 个随机种子；报告中位数、最差 seed、bootstrap 95% CI。
- smoke 可用 20 个随机对照，仅验证代码；正式判定至少 99 个表达量/FC/检测率/token rank 匹配的 null，使用 `(k+1)/(n+1)`，跨候选做 BH-FDR。
- null 可按匹配分层预计算并复用，减少重复 GPU 计算；复用规则写入 manifest。

正式 `PASS` 必须同时满足：

1. 两路径均 `evaluable`；
2. 两路径 `median(rescue_excl_target) > 0`；
3. 每路径严格多数 donor 同方向，即 `donor_consistency > 0.5`，且 `evaluable_donors >= 3`；
4. 3 seeds 中至少 2 个方向一致，最差 seed 不出现强反向效应；
5. KO 主结论为 `mask`，三种 KO mode 至少 2/3 同方向；只在 `delete` 有效则不通过；
6. 双路径候选层 `q_value < 0.05`；
7. 未扰动模型质量门通过。

任一硬门不满足但数据齐全为 `FAIL`；数据/资产/token/donor/运行异常导致不能计算为 `INCONCLUSIVE`。

---

## 5. 测试与技术指标

### 5.1 测试矩阵

| 层级 | 位置 | 内容 | 门禁 | 目标耗时 |
|---|---|---|---|---|
| T1 纯单元 | `tests/unit/integration/perturbgen/`、模型单测 | contracts、preflight、resolver、config、parser、statistics、manifest | 每个 PR | 3–5 min |
| T2 mocked 集成 | `tests/integration/test_perturbgen_*_mocked.py` | 六 stage 顺序、错误传播、resume、报告、批量三态 | 每个 PR | 5–8 min |
| T3 真环境 smoke | `tests/real_assets/test_real_perturbgen_smoke.py` | 真 tokenise/短训/perturb/export/output schema | 独立 opt-in/nightly | 15–30 min，实测回填 |
| T4 科学验收 | 固定冻结 cohort + evidence JSON | held-out donor、3 seeds、双路径、null、mode、FDR | release gate | 依数据/GPU 实测 |
| T5 Gate-E | 固定 PTM→gene 基准集 | 词表迁移、DAVF 重训、旧/新配对指标 | 删除 Geneformer 前 | 2–6 h，实测回填 |

### 5.2 单元与回归必测项

| 模块 | 正常 | 边界/异常 | 回归 |
|---|---|---|---|
| data_prep | 合法 counts/ENSG/obs | log-only、重复 ENSG、单 donor、零库 | 不把 `X` 静默当 counts |
| resolver | ENSG/symbol 命中 | 歧义 symbol、缺失、重复 token | 未知基因不哈希 |
| config | src/tgt YAML | 非法 tp/mode/filter | count decoder 路径不误写成 encoder |
| runner | 六 stage 成功 | timeout、非零码、路径逃逸、脏输出 | export stage 不遗漏；不用 `shell=True` |
| manifest | 指纹一致复用 | 数据/seed/config/ckpt 任一变化 | 禁止“文件存在即 resume” |
| results | R 正/负/零 | 空 signature、shape 不齐、少 donor | 目标基因大跌但其他不变不得通过 |
| verdict | 双正 PASS | 单负 FAIL、一路不可评估 INCONCLUSIVE | 严格 `>0.5`，不是 `>=0.5` |
| embedding | 真实资产读取 | hash/维度/token 行不符 | 不生成随机 embedding |
| architecture | 默认构造与新资产构造 | 缺资产、旧 checkpoint | runtime 主链确实使用新矩阵 |

### 5.3 工程指标

| 指标 | 验收标准 |
|---|---|
| 主环境隔离 | `python -c "import src.integration"` 不 import PerturbGen；core CI 无 PerturbGen 依赖；新增 analysis job 安装主进程所需 anndata |
| 配置正确性 | 锁定 commit 的 golden YAML 全字段匹配；未知字段在本地 schema 阶段报错 |
| 结果解析 | 固定 10k cells workload 的 P95 < 30 s；峰值 RSS 相对 M0 冻结基线回归不超过 15%，实际硬件写入报告 |
| runner 开销 | mock 子进程外调度 P95 < 1 s/任务 |
| GPU 性能 | M0 建立固定 workload 基线；后续单 perturb P50/P95、完整候选/小时回归不超过 15% |
| 峰值显存 | A100 参考软门：单 perturb < 12 GiB、export < 16 GiB；其他 GPU 先实测再冻结门槛 |
| 稳定性 | 同 seed/同指纹报告哈希一致；3 seeds 方向一致性按 §4.7 |
| 磁盘 | 启动前通过 2×估算 + 10 GiB 空间门；每个 artifact 有 hash/size/retention 字段 |
| CI | T1+T2 全绿；主项目 branch coverage 不低于现有 74% 门 |

绝对吞吐不能在未知 GPU、细胞数、基因数和 mode 数时预设。M0 输出基线后，把硬件型号、CUDA、batch、细胞数、基因数、双路径/mode 组合写入 benchmark manifest，再冻结 release 门。

### 5.4 科学质量门

| 门 | 标准 |
|---|---|
| 未扰动预测 | 3 seeds 的 held-out DEG 方向恢复中位数 ≥0.60，最差 seed ≥0.50；signature 相关性 >0 |
| donor | 每路径可评估 donor ≥3；严格多数同方向 |
| null | 正式分析 ≥99 匹配 null；候选层 BH-FDR `q<0.05` |
| mode | mask 为主；mask/pad/delete 至少 2/3 同方向 |
| 双路径 | 两路径分别过门后再做 AND；不允许用合并均值掩盖单路径失败 |
| Gate-E 数据 | 至少 200 个可追溯 PTM→gene 样本，固定划分，bootstrap 95% CI |
| Gate-E 词表 | 基准 gene 覆盖率 ≥99%，token collision=0，PTM action code 在迁移前后 100% 不变 |
| Gate-E DAVF | 主下游指标相对旧冻结基线下降 ≤1 个百分点，且 95% CI 不显示明显劣化 |

Gate-E 比较的是语义解析覆盖和 DAVF 下游表现，不能比较旧/新数字 token ID 是否相等，也不能把 mapper 固定规则的“方向一致率”当 embedding 质量。

### 5.5 本轮实际验证基线

```text
pytest -q --disable-warnings \
  tests/unit/test_davf.py \
  tests/unit/test_davf_inference.py \
  tests/unit/test_ptm_direction_mapper.py \
  tests/unit/models/test_geneformer_embedding.py

结果：71 passed，7 warnings，7.13 s。
```

另一次含 GenKI/集成用例的 147 项命令在 120 秒超时，输出已显示上述四组单元测试通过并开始执行集成测试；该次不计为全通过。项目历史基线为 2026-08-20 的 1957 单元 + 107 集成通过，本轮未重新执行全量套件。

---

## 6. 风险清单与应对

| 风险 | 概率/影响 | 早期信号 | 应对与阻断门 |
|---|---|---|---|
| 静态 token embedding 无稳定 checkpoint 键 | 中/高 | M0 无法唯一定位矩阵或行号 | M0 直接阻断工作流 B；不得猜 key 或从样本 embedding 代替 |
| 词表行号与模型 embedding 行错位 | 中/高 | known ENSG round-trip 失败 | 导出时随机抽样 + 全量 hash/row 校验；collision 必须为 0 |
| mapper 方向语义继续混用 | 高/高 | action code 被写成 up/down | 类型拆分 `observed_direction/davf_action`；静态类型和单测阻断 |
| DAVF runtime 未实际使用新 embedding | 中/高 | 只改 loader，输出不随资产变化 | architecture 注入测试 + 参数 provenance；删除死配置 |
| 新底座性能下降 | 中/高 | Gate-E CI 劣化 | 不删除旧版本；停止发布并回到方案层重议，禁止运行时随机回退 |
| PerturbGen inference 首次拉 rouge 失败 | 高/中 | 离线环境初始化报错 | 镜像构建时预热/缓存 evaluate module；M0 做离线 smoke |
| normal source 无目标表达 | 高/中 | 检测率/有效 cell 为 0 | 标 `INCONCLUSIVE`；不把阴性当靶点无效；正式 AND 不通过 |
| donor 混杂或不足 | 中/高 | 状态仅由 1–2 donor 支撑 | preflight 阻断正式分析；只允许探索报告 |
| signature 数据泄漏 | 中/高 | 同 donor 同时建 signature 和验收 | donor-held-out 划分；manifest 记录 donor 集合 |
| 20 null 的假显著 | 高/高 | p 值只能到约 0.0476 | smoke/正式分层；正式 ≥99 + add-one p + BH-FDR |
| 多 seed/mode/null 计算量爆炸 | 高/高 | 任务数超预算 | 先 mask screening，再对入围者跑敏感性；匹配 null 分层复用；dry-run 输出任务数/预计资源 |
| GPU 并发 OOM | 中/高 | 多子进程同卡、显存逼近上限 | 默认串行 GPU lock；显存门；不自动缩 batch 掩盖配置问题 |
| h5ad 磁盘膨胀 | 高/中 | 控制组/seed/mode 产物成倍增长 | 运行前容量估算；artifact retention 分级；默认不自动删除 |
| 上游 commit/schema 漂移 | 中/中 | golden config 或输出字段变化 | 锁 commit；升级必须独立变更单和契约测试 |
| resume 使用陈旧资产 | 中/高 | 路径存在但配置已变 | 全指纹 manifest；任何不一致拒绝复用 |
| 子进程路径/命令注入 | 低/高 | 用户路径越界或含 shell 字符 | resolve + allow-root、参数数组、禁止 shell=True、日志脱敏 |
| CI 只绿 mocked 测试 | 高/高 | real smoke 长期未运行 | 独立 real-assets/GPU workflow；release evidence 缺失即阻断 |
| 结果被解释为因果或临床证据 | 中/高 | 报告省略模型/数据限制 | 报告固定免责声明；只称计算候选，必须后续实验验证 |
| 维护两个外部环境 | 高/中 | lock/驱动/缓存漂移 | 输出 lockfile/container digest/SBOM；季度升级，不跟随上游自动漂移 |

---

## 7. 可直接执行的阶段计划

### 7.1 排期口径

`T0` 是“方案批准 + encoder 权重可读 + 一份满足契约的队列样本 + GPU 环境可用”的首个工作日。若 T0 为 2026-08-24，基准排期如下；任何外部资产延迟按工作日顺延，不把等待时间算作已完成。

| 阶段 | 基准日期 | 人天 | 依赖 | 里程碑 |
|---|---:|---:|---|---|
| M0 可证伪试验 | 08-24～08-25 | 2 | 权重、独立环境 | Gate-0 |
| M1 契约与数据 | 08-26～08-28 | 3 | 队列 schema | Gate-1 |
| M2 runner 与配置 | 08-31～09-03 | 4 | M0 部分结论 | Gate-2 |
| M3 双路径与统计 | 09-04～09-09 | 4 | M1/M2 | Gate-3 |
| M4 DAVF 底座迁移 | 09-10～09-16 | 5 | Gate-0/M2 | Gate-E |
| M5 CI/性能/真实 smoke | 09-17～09-21 | 3 | M3/M4 | Gate-4 |
| M6 科学验收 | 09-22～10-02 | 3–8 | 冻结真实队列/GPU | Gate-5 |
| M7 收口与移除 Geneformer | 10-05～10-06 | 2 | Gate-E/Gate-5 | Release |

总量：26–31 人天，基准日历约 6 周。PerturbGen 训练、99+ null 和 3 seeds 的 GPU 墙钟时间需在 M0/M3 dry-run 后实测回填；在此之前不承诺完成日期。

### 7.2 各阶段 `<files>/<action>/<verify>/<done>`

#### M0：真实权重与外部契约试验

- **files**：临时试验脚本（不进生产包）、`outputs/perturbgen/spike/<run_id>/evidence.json`。
- **action**：建立独立环境；验证离线 inference 初始化；定位实际 token embedding；校验 token dict→row；跑最小官方 perturb config；测一次耗时/显存/输出大小。
- **verify**：已知 ENSG round-trip；embedding shape/hash；h5ad schema；离线 rouge 缓存；GPU/磁盘证据。
- **done / Gate-0**：六项全部有真实 evidence。任一核心项失败则工作流 B `BLOCKED`，先改方案，不进入 DAVF 代码改造。

#### M1：契约、数据和候选证据

- **files**：`contracts.py`、`data_prep.py`、`gene_vocabulary.py`、对应单测/fixture。
- **action**：实现三态、raw counts/ENSG/donor/state 校验、symbol↔ENSG、表达筛查、observed direction 与 DAVF action 分离。
- **verify**：正常/空/错误/回归用例；至少 3 donor 门；hash fallback 回归用例必须失败。
- **done / Gate-1**：T1 相关测试全绿，真实队列 preflight 报告无未解释硬错误。

#### M2：外部环境调度与可恢复执行

- **files**：`env_guard.py`、`config_builder.py`、`runner.py`、embedding export 两个脚本、配置模板、mocked integration。
- **action**：实现六 stage、GPU lock、timeout、路径白名单、全指纹 manifest、dry-run 任务/资源估算。
- **verify**：golden YAML；exit/timeout/路径逃逸/脏输出/陈旧 resume；真 export/perturb smoke。
- **done / Gate-2**：T1/T2 全绿；同指纹可复用、任一指纹变化拒绝复用。

#### M3：双路径、rescue 和报告

- **files**：`results.py`、`dual_path.py`、`reports.py`、CLI、统计/集成测试。
- **action**：held-out signature、排目标基因、donor/seed/mode/null/FDR、PASS/FAIL/INCONCLUSIVE、artifact 报告。
- **verify**：已知答案合成数据；目标基因假阳性；单路径负；一路不可评估；20 null 只标 smoke。
- **done / Gate-3**：严格 AND 全部回归通过；报告可从 manifest 完整重放。

#### M4：DAVF 底座迁移

- **files**：`architectures.py`、`davf_inference.py`、`latent_davf.py`、`davf.py`、`ptm_direction_mapper.py`、DAVF 配置/训练脚本/测试。
- **action**：接入公共 resolver 和真实 PerturbGen matrix；移除死配置；版本化 checkpoint；重训 DAVF；运行 Gate-E。
- **verify**：runtime 确实读取新资产；覆盖/碰撞/action code；≥200 基准样本；下游非劣。
- **done / Gate-E**：§5.4 全部满足。未通过时保留旧发布版本并停止迁移，不启用运行时随机 fallback。

#### M5：自动门禁、性能和真实 smoke

- **files**：pytest markers、real-assets tests、独立 workflow、benchmark 脚本。
- **action**：PR 跑 T1/T2；nightly/手工跑 T3；记录 P50/P95、显存、RSS、磁盘；冻结后续回归阈值。
- **verify**：从干净环境执行；evidence JSON 上传；故意破坏资产确认 release gate 会失败。
- **done / Gate-4**：mock 绿与真环境绿同时存在；无 evidence 不允许发布。

#### M6：冻结队列科学验收

- **files**：冻结 cohort manifest、候选 manifest、统计报告、原始配置和日志。
- **action**：3 seeds、held-out donors、两路径、KO 敏感性、≥99 matched null、BH-FDR；复核资源预算。
- **verify**：独立重算报告；bootstrap CI；审查 donor 与候选泄漏。
- **done / Gate-5**：每个候选有完整三态和原因；只有双路径全过者进入实验验证清单。

#### M7：收口、文档和移除 Geneformer

- **files**：README/使用指南、依赖契约、Geneformer 文件及专属测试、release evidence。
- **action**：Gate-E 与 Gate-5 通过后删除 Geneformer 主链/资产说明；更新配置迁移指南；archive 方案。
- **verify**：`rg -n "geneformer" src configs scripts tests` 仅允许明确历史/迁移引用；全量测试、lint、mypy。
- **done / Release**：全量门禁绿、evidence 可追溯、文档状态从“待审核”改为“已实施归档”。

### 7.3 阶段停止条件

以下任一情况必须停止当前阶段并回到方案审核，不得打补丁绕过：

1. 无法从真实 checkpoint 唯一导出 token matrix；
2. 词表与 embedding row 无法一一对应；
3. 真实数据少于 3 个可评估 donor 或只有 log 数据；
4. 未扰动质量门不通过；
5. Gate-E 显著劣化；
6. 外部 schema 与锁定 commit 不一致；
7. GPU/磁盘预算超过已批准资源；
8. 任何实现需要修改 PerturbGen 上游源码。

---

## 8. 预期效果

| 结果 | 可预期程度 | 验证方式 |
|---|---|---|
| 形成 normal→disease 早期干预与 disease within-state 治疗干预两条可重放流水线 | 高 | T2/T3 + manifest 重放 |
| 消除 Geneformer 未知基因哈希映射和随机 embedding fallback | 高 | resolver collision=0、缺资产 fail-fast |
| 统一 PerturbGen/DAVF 基因词表和资产 provenance | 高（Gate-0 通过后） | token round-trip、manifest |
| 降低假阳性候选 | 中高 | 排目标基因、双路径 AND、donor/seed/null/FDR |
| DAVF 下游指标提升 | 未知 | Gate-E 配对实验；只要求先非劣，不预设提升 |
| 产生可直接实验验证的高置信候选 | 中，依赖真实数据与模型质量 | Gate-5；报告仍只属于计算证据 |

---

## 9. 实施前必须提供或确认

| 项 | 最小要求 | 缺失影响 |
|---|---|---|
| PerturbGen 独立环境 | Python 路径、GPU/CUDA、锁文件或容器 digest | M0/M2/M5 阻塞 |
| encoder 权重 | 用户手动提供，只读路径 + SHA-256 | M0/M4 阻塞 |
| normal/disease AnnData | raw counts、ENSG、cell type/state/donor，至少 3 可评估 donor | M1/M3/M6 阻塞 |
| DAVF 训练数据/checkpoint | 旧基线、训练入口、固定 PTM→gene 基准集 | M4/Gate-E 阻塞 |
| 计算与存储预算 | GPU 型号/数量、可用时段、磁盘额度 | 排期和正式 null 数无法冻结 |

本方案批准只授权按阶段实施，不授权自动下载大权重、不授权修改 PerturbGen 上游源码、不授权把长任务接入同步 API，也不授权绕过任一 Gate。

---

**审核状态**：v2.0 已获用户实施授权；当前执行状态与阻断项见 §10。

---

## 10. 2026-08-22 实施记录

| 阶段 | 当前状态 | 已落地证据 | 未完成门槛 |
|---|---|---|---|
| M0 | **BLOCKED** | 已确认源码 commit `a9a9375`、GPU Tesla P40；重新审计本地 30 个 scPerturb h5ad：26 个可读、4 个截断，唯一有显式 `patient` 的 Shifrut 队列仅 2 人且全为 healthy；`outputs/perturbgen/spike/20260822_reassessment/evidence.json` 保存明细 | 缺 encoder 权重、独立 Python 3.11 环境、满足 normal/disease 配对与 ≥3 donor 的合规 cohort；未验证 tensor key/离线 inference/真实 perturb |
| M1 | 工程实现完成 | contracts、raw-count/ENSG/donor preflight、严格 resolver、source-KO 零表达 `INCONCLUSIVE` | 真实 cohort preflight 尚未执行 |
| M2 | mocked 门完成 | 六 stage runner、上游下划线 CLI、路径约束、目录树 hash、输出 hash resume、GPU lock、磁盘门、src/tgt 双计划；tokenise 六个产物按上游公式精确绑定；动态 checkpoint/h5ad 按隔离目录严格唯一发现；manifest 记录具名产物路径/hash，下游显式引用 | T3 真实 PerturbGen 环境仍未执行；真实产物契约仍须 Gate-0 证据确认 |
| M3 | 工程实现完成 | held-out donor signature、排目标基因 rescue、3 seeds、mode、99+ null、BH-FDR、未扰动质量门、严格 AND；新增版本化评估 CLI，严格从成功 stage manifest 绑定的 h5ad `pred_counts`/`X` 按 donor 切片聚合并生成候选报告 | 尚未用真实 PerturbGen 产物执行候选评估 |
| M4 | **未开始** | 已实现严格 embedding 导出/加载资产协议，不猜 key、不允许 hash/random fallback | Gate-0 未通过，禁止接入 DAVF runtime、重训或执行 Gate-E |
| M5 | 自动门禁已完成，Gate-4 未通过 | 新增独立 self-hosted GPU workflow、正式 release evidence 硬门和 benchmark CLI；每个正式样本必须双路径成功、含两个独立 h5ad、峰值显存采样及 P50/P95 耗时/RSS/显存/产物体积；证据缺失、过期、skip 或指标缺失均失败 | 真实资产/GPU 测试当前因 Gate-0 缺失而 skip；尚无真实 evidence，不得宣称 Gate-4 |
| M6–M7 | 未开始 | 无 | 依赖 M4、真实 smoke、科学验收与 Gate-E/Gate-5 |

本轮验证：新增/相关 DAVF 聚焦套件 `146 passed`，ruff 全绿；最终 PerturbGen 生产代码、脚本、单元/模拟集成/真实资产入口聚焦回归为 `111 passed, 1 skipped`，skip 是未提供真实资产时的预期行为；M3/M5 收口子集另为 `32 passed, 1 skipped`。全仓 `2289` 项测试在 600 秒上限内仅运行到约 3%，命令超时退出（exit 124），因此不记为全量通过。

本记录不改变 §7.3 停止条件。尤其不得把 mocked 通过解释为 Gate-0、Gate-E 或真实双路径科学验收通过。
