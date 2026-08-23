# PTM2CellNet 项目综合分析报告（2026-08-23）

## 0. 任务判断与输入依据

本任务类型：Bug 修复、技术债清理、运行环境建设、端到端（E2E）状态系统性复核。

依据用户指令，本轮以日期最新的修复报告 `[R3]`（`project_repair_report_20260822.md`）§5「严重/阻塞问题与后续策略」为修复依据，重点解决其中排期在 **2026-08-23 时间窗口**的两项：

1. **独立运行环境缺失**（`[R3]` §5 第一行：建立 Python 3.11 的独立 PerturbGen 环境，锁定源码 commit、CUDA/GPU、依赖版本，不改当前混杂环境）；
2. **Geneformer 残留**（`[R3]` §4.2 与 `[R1]` §3.2 A2/A3/A5：`davf_inference.py` 的 `geneformer_path` null→随机 embedding 死语义、`architectures.py` 死配置透传、`configs/davf_integration.yaml` 旧 schema、`latent_davf.py` 注入链路缺失、`finetune_davf_e2e.py` 无资产参数）。

引用文档：

| 标记 | 文档 | 重点章节 |
|---|---|---|
| `[R1]` | `project_analysis_20260822.md` | §1.3、§3.2（A2/A3/A5）、§4.4、§5.2、§7.2 |
| `[R2]` | `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` | §4.5、§5.4、§7.2（M4/M7）、§7.3 |
| `[R3]` | `project_repair_report_20260822.md` | §2、§4.2、§5、§6 |
| `[CS]` | `docs/CURRENT_STATUS.md` | Quick Reference 全节 |

---

## 1. 问题修复汇总（按严重程度分类）

### 1.1 严重（阻塞级）——独立运行环境缺失（`[R3]` §5 2026-08-23 项）

**修复状态：✅ 完全解决（含 GPU 验证与 token round-trip，详见 §2 迭代一）。**

| 项目 | 结果 |
|---|---|
| Python 3.11 独立环境 | ✅ conda env `perturbgen`，Python **3.11.15**（与主环境 SSH_unit 3.12.13 完全隔离） |
| 权限路径 | 本机 conda/uv 可用；docker 与 sudo 需密码不可用——**未触发"写脚本请用户手动执行"分支**，conda 用户目录建环境无需提权 |
| 源码 commit 锁定 | ✅ `ref/Perturbgen-src` @ `a9a937526574b70aaa5fbf592bb56f35765ca83c`（2026-07-22），editable 安装 `PerturbGen-0.1.0` |
| GPU/CUDA 记录 | ✅ Tesla P40 24GB、driver 580.173.02；**torch 2.5.1+cu124 实测 `cuda_available=True`，真实 checkpoint 在 cuda:0 加载并执行 GPU matmul** |
| 依赖版本锁定 | ✅ 按上游 `pyproject.toml` 锁定版全部安装（torch 2.5.1 / pytorch-lightning 2.4.0 / transformers 4.47.0 / tokenizers 0.21.0 / datasets 3.0.0 / evaluate 0.4.3 / huggingface-hub 0.26.5 / safetensors 0.4.5 / scanpy 1.11.2 / anndata 0.11.0 / numpy 2.2.0 / pandas 2.2.3 等 88+ 包） |
| HF/evaluate 离线初始化 | ✅ `HF_HUB_OFFLINE=1` 等三变量下 transformers/datasets/evaluate/tokenizers/safetensors/anndata/scanpy/h5py/pytorch_lightning 全部 import 成功（evidence JSON `offline_imports_ok=true`） |
| M0 token round-trip | ✅ 独立环境加载上游 `token_dict_gftokens_gc95M.pkl`（18967 项），ENSG00000187634/188976/187961 → token id 4/5/6 |
| CLI 入口 | ✅ `perturbgen --help` 四子命令（tokenise / train-mask / train-decoder / extract-embedding）正常 |
| Evidence 文件 | ✅ `outputs/perturbgen/env_evidence_20260823.json`（python/commit/GPU/torch CUDA/pip freeze/离线 import 全记录） |
| 复现脚本交付 | ✅ `scripts/setup_perturbgen_env.sh`（五步全自动）+ `scripts/download_perturbgen_wheels.sh`（不稳定网络下的 curl 断点续传 + sha256 校验下载器） |

**关键发现（自主分析结论）**：`[R3]` 与上游 `perturbgen/requirements.txt` 中的 `-e git+ssh://git@github.com/amirvhd/T_perturb.git@...` **不是 Python 包依赖**——`T_perturb` 只是目录名（`perturbgen/configs/paths.py` 用 `ROOT / "T_perturb"` 做路径拼接；train/val 中的 `./T_perturb/...` 仅为默认路径字符串；全库无 `import T_perturb`）。因此私有 SSH 仓库不可得**不阻塞**环境搭建，editable 安装 `pip install --no-deps -e .` 即可。`pyproject.toml`（poetry 版）本身已无此依赖。同理 `mpi4py` 在 `perturbgen/`、`scmaskgit/` 中零直接 import（仅 Dockerfile 为保险装 libopenmpi-dev），M0 推理闭包不需要它。

### 1.2 严重——Geneformer 死语义与注入链路缺失（`[R1]` §3.2 A2、§4.2、`[R3]` §4.2）

**修复状态：代码层全部完成，含真实资产验证。** 依据 `[R2]` §4.5「修改文件与兼容策略」逐项落地：

| 文件 | 修改 | 对应方案条款 |
|---|---|---|
| `src/models/davf_inference.py` | ① 删除 `geneformer_path` 字段、docstring 与「null → random embeddings」警告（原 :70/:76/:123-126）；② 新增 `embedding_asset_path` 配置，构造时经 `load_perturbgen_embedding_asset()`（sha256+schema 严格校验）加载 PerturbGen 冻结矩阵，传入 `LatentDAVF(pretrained_gene_embeddings=...)`——**打通 `[R1]` §4.2 指出的「注入链路缺失」**；③ asset 缺失/损坏直接抛 `PerturbGenEmbeddingError`，不降级；④ checkpoint 加载新增 `schema_version` 显式版本检查（≠1 即报错，不静默 zero-fallback） | §4.5 davf_inference 行：「删除死的 geneformer_path 语义；向 LatentDAVF 传预训练矩阵/词表」「checkpoint schema 显式版本化，不做 silent partial load」 |
| `src/models/architectures.py` | ① `DAVFConfig` TypedDict 删除 `geneformer_path`，新增 `embedding_asset_path`；② 删除两处无意义自赋值透传（原 :566/:630）；③ `__init__` use_davf 分支新增 **schema v2 迁移闸门**：任何来源（yaml/dict/from_config）的 `davf_config` 含 `geneformer_path` 即抛 `ValueError` 并给出迁移指引——单一 chokepoint；④ `DAVFInferenceConfig` 构造透传 `embedding_asset_path` | §4.5 architectures 行 + 迁移段「发布版采用配置 schema v2，一次性移除 geneformer_path，提供显式迁移检查/报错」 |
| `configs/davf_integration.yaml` | 升级 **schema v2**：移除 `geneformer_path: null`，新增 `embedding_asset_path: null`（注释说明指向 PerturbGen 资产目录、fail-fast 语义、导出器入口） | §4.5 配置行：「不提供随机 fallback 配置」 |
| `src/models/davf_checkpoint_utils.py` | `save_checkpoint` 写入 `schema_version: 1`；`load_checkpoint` 读取端校验未知版本即报错 | §4.5「checkpoint schema 显式版本化」 |
| `scripts/finetune_davf_e2e.py` | 新增 `--embedding-asset` CLI 参数并接入 `build_model()`（`[R1]` §3.2 A3 项）；`scripts/finetune_davf.py` 走 `from_config` + config 子键路径，schema v2 下自动获得资产注入能力，无需改码 | §4.5「scripts/finetune_davf.py 及 E2E 脚本：新词表/embedding/manifest 参数」 |
| 测试同步 | `tests/unit/test_architecture_davf_integration.py`（旧契约断言 :112/:119/:321/:329/:343 全部改为 schema v2 契约 + 新增迁移闸门回归 + 临时真实 asset fixture）；`tests/unit/test_davf_inference.py` 新增 `TestEmbeddingAssetInjection`（5 项：字段级死语义禁复活、矩阵逐元素注入、缺失 asset fail-fast、gene 模式隔离、未知 schema_version 在 strict 模式下 forward 报错） | §4.5 迁移段「必须同步修改 test_architecture_davf_integration.py…及所有读取 geneformer_path 的配置断言」 |

**真实资产验证（超出单测的实测证据）**：使用 `[R3]` §2 第三轮导出的真实资产 `outputs/perturbgen/embedding_asset_20260822/` 构造 `DAVFInferenceModule(embedding_asset_path=...)`，实测：`gene_embed_table` 为 `(18967, 768) float32`，自动构建 `Linear(768→192)` 投影，首行（ENSG00000187634）与末行（第 18966 行）与资产矩阵**逐元素相等**（`torch.equal` True）。

### 1.3 明确保留边界（非残留，不清理）

依据 `[R2]` §7.3 停止条件（Gate-0 未过不动 runtime 默认链路）与 §7.2 M7（Gate-E/Gate-5 通过后才移除主链），以下 `geneformer` 引用**有意保留**：

| 位置 | 性质 | 处置 |
|---|---|---|
| `src/models/geneformer_embedding.py`（67 处） | 现网 DAVF 主链 loader（`PTMDirectionMapper` 单例默认使用）+ legacy checkpoint 兼容 | M7 处理；其 hash fallback 已有 `PTM2CELLNET_STRICT_MODEL_ASSETS=1` fail-fast（`[R1]` §5.2） |
| `src/models/ptm_direction_mapper.py`（15 处） | PTM→gene 方向映射主链 | M4 完整接入时换公共 resolver |
| `src/models/{davf,latent_davf,legacy_davf,biperturb,delta_predictor}.py` | 注释性历史说明（如「Geneformer 初始化嵌入表」）+ legacy 兼容层 | 历史说明属 `[R3]` §5 M7 允许的「历史说明或明确兼容边界」 |
| `configs/integration/perturbgen.yaml:67` `src_mode: "Geneformer"` | **PerturbGen 上游 tokenisation 模式契约值**（`config_builder.py:291-292` 强制校验），非本项目 Geneformer 依赖 | 保留，上游 API 值 |

清理后量化对比：`davf_inference.py` 67→**1**（docstring 历史说明）、`architectures.py` 3→**3**（全部为迁移闸门必要代码）、`configs/davf_integration.yaml` 2→**2**（schema v2 变更注释）。运行时语义引用清零。

---

## 2. 每轮迭代的执行情况与结果

### 迭代一：独立 PerturbGen 环境（完成）

1. **环境探查**：`python3.11` 系统不存在；conda/uv 可用；docker daemon 无权限（需 sudo 密码）；GPU Tesla P40 24GB（driver 580.173.02）；PerturbGen 源码为 git repo（commit `a9a9375`）。
2. **依赖闭包分析**（避免盲目装 poetry 全家桶）：通读 `pyproject.toml`、根/包级 `requirements.txt`、`Dockerfile`、`perturbgen/__main__.py` 与 `Perturb/`、`pp/` 全部 import——确认 M0 推理闭包 = torch/lightning/transformers/datasets/evaluate/scanpy/anndata/h5py/loompy/click/wandb/sympy；排除 mpi4py/jax/deepspeed/ray/T_perturb（依据见 §1.1 关键发现）。
3. **conda create -n perturbgen python=3.11**：成功，Python 3.11.15。
4. **不稳定网络三阶段攻坚**：① 直接 pip 下载在 torch 519MB/906MB 处因 SSLEOFError 六次重试后崩溃（partial 文件随进程消亡）；② 加 `--resume-retries 200` 重跑，续传至 cudnn 664MB/678MB 处再次因 ProtocolError 崩溃且不走 resume 分支；③ **自研持久化下载器** `scripts/download_perturbgen_wheels.sh`：pip dry-run report 解析全部 95 个 URL+sha256 → `curl -C -` 字节级断点续传 + 每文件 500 次重试循环 → sha256 逐文件校验。下载全程约 3 小时（128–400 KB/s 波动、多次断连均无损续传），**95/95 文件 sha256 全部验证通过**（校验前缀 bug `sha256=` vs `sha256:` 已修正）。
5. **离线安装**：`pip install --no-index --find-links=~/perturbgen_wheels`（88+ 包一次成功；loompy 为 sdist 需 build 依赖，单独在线补装）+ `pip install --no-deps -e .`（PerturbGen-0.1.0 editable）。
6. **四重实测验证**：① evidence 采集 `overall=OK`（python 3.11.15 / commit / GPU / cuda_available=True / 9 模块离线 import 全过）；② 真实 checkpoint GPU 加载（162 keys，`transformer.token_embedding.weight` (19003, 768) 上 cuda:0，GPU matmul 真实执行）；③ token round-trip（pickle 词表 18967 项，3 个 ENSG → token 4/5/6）；④ CLI 四子命令正常。

### 迭代二：Geneformer 残留清理（完成）

按 §1.2 表格逐文件实施；中途发现直接透传虚构路径的旧测试与 fail-fast 语义冲突（构造即加载校验），按「正式启用时资产缺失 fail-fast」（`[R2]` §4.5）原则修正测试为临时真实 asset fixture，而非放宽生产代码——**保留 let-it-crash 语义，不注入兜底**。

### 迭代三：回归与静态门禁（完成）

| 检查 | 命令 | 结果 |
|---|---|---|
| DAVF 聚焦单元 | `pytest tests/unit/test_davf_inference.py tests/unit/test_architecture_davf_integration.py tests/unit/test_davf_core.py` | **78 passed** |
| DAVF 全量相关 | 离线变量 + `pytest test_davf{,_foundation_regressions}.py test_legacy_davf_compat.py test_finetune_davf_script.py test_ptm_direction_{mapper,v22}.py tests/integration/test_davf_pipeline.py tests/unit/models/test_perturbgen_embedding.py` | **82 passed, 1 skipped** |
| finetune 脚本 | `pytest tests/unit/test_davf_e2e_finetune.py tests/unit/test_finetune_davf_script.py` | **16 passed** |
| 全量套件 | `unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest tests -q` | **2321 passed, 16 skipped, exit 0, 520.53s**（上轮基线 2315 passed `[R3]` §4.1，本轮 +6 为新增测试，零失败） |
| 编译 | `python -m compileall -q src scripts tests` | 退出码 0 |
| Lint | `ruff check src scripts tests` | All checks passed |
| 类型 | `mypy`（改动三文件） | 0 新增错误（5 errors 全在 `src/api/routes/cross_scale.py`，为 `[CS]` 记录的既有基线） |
| 真实 asset 注入 | 见 §1.2 实测段 | 首末行逐元素相等 |

---

## 3. 系统性复核分析：E2E 训练与推理技术要求

### 3.1 主进程（PTM2CellNet 自身）E2E：**满足工程要求**

- **训练链路**：`scripts/train.py`、`train_lightning.py`、`finetune_davf*.py` 等 CLI 入口实测可用（本轮 smoke：`--help` 正常解析，mamba_ssm deprecation warning 为已知无害项）；训练基础设施（Trainer/AMP/grad-clip/checkpoint/resume）由 2321 项测试覆盖（合成数据口径，`[CS]`「解释测试结果时的边界」）。
- **推理链路**：`predict.py`/`batch_predict.py`/`predict_variant_effect.py` + FastAPI `/predict`/`/batch_predict`/`/model/info` 等路由正常；`use_davf` 路径在 schema v2 下行为不变（默认 `embedding_asset_path: null` = 训练态嵌入表 + checkpoint 权重，`[R2]` §4.5「默认不开启新链路时行为不变」达成）。
- **量化**：全量 2321 passed / 0 failed；ruff/compileall 通过；mypy 维持 23 基线（本轮 0 新增）。

### 3.2 PerturbGen 独立链路 M0 六项 evidence（对照 `[R3]` §4.2 / `[R2]` §7.2 M0）

| M0 evidence | 上轮状态 | 本轮后状态 |
|---|---|---|
| ① 真实 encoder checkpoint + tensor key | ✅（`[R3]` §2 第三轮） | ✅ 并新增独立环境 GPU 加载证据（162 keys / (19003,768) / cuda:0） |
| ② token↔row 对齐 | ✅ 静态验证 | ✅ 并新增独立环境 token round-trip（3 ENSG → 4/5/6） |
| ③ Python 3.11 独立环境 | ❌ 阻塞 | ✅ **完成**（3.11.15 + commit/GPU/依赖锁定 + evidence JSON + 复现脚本） |
| ④ 离线初始化/inference | ❌ | 🟡 **初始化完成**（9 模块离线 import + GPU ckpt 加载 + CLI）；真实 perturb inference 需 tokenized 数据 |
| ⑤ 真实 perturb smoke + 资源基线 | ❌ | ❌ 依赖真实数据（M0 窗口 08-24～25 执行） |
| ⑥ 合规 normal/disease donor 队列 | ❌ 阻塞 | ❌ **仍阻塞**（外部数据依赖：需 raw counts/ENSG/donor 标注的 AnnData，≥3 可评估 donor；本地 30 个 h5ad 审计结论不变 `[R3]` §4.2） |

### 3.3 DAVF × PerturbGen 集成链路（M4 代码前置）

| M4 子项（`[R2]` §4.5） | 状态 |
|---|---|
| 删除 `geneformer_path` 死语义 | ✅ 本轮 |
| LatentDAVF 预训练矩阵注入链路 | ✅ 本轮（真实 18967×768 验证） |
| 配置 schema v2 + 显式迁移报错 | ✅ 本轮 |
| checkpoint schema 显式版本化 | ✅ 本轮（v1 写入+校验；完整 strict-load 待重训） |
| finetune/E2E 脚本资产参数 | ✅ 本轮（`--embedding-asset`） |
| runtime 默认链路切换到 PerturbGen 资产 | ⛔ **有意不做**——Gate-0 未过（`[R2]` §7.3），默认行为保持不变 |
| DAVF 重训 + Gate-E（≥200 基准） | ❌ 未开始（依赖 M0-M3 真实资产与队列） |

### 3.4 结论

> 主进程工程 E2E（合成数据口径）**全部满足**；PerturbGen 独立环境 M0 ①②③④ 已闭环（环境/资产/对齐/初始化），剩余阻塞收敛为**真实数据类缺口**（M0⑤ 真实 perturb smoke、M0⑥ donor 队列）；DAVF 底座迁移的**代码侧前置全部就绪**，Gate-0 数据条件满足后即可按 `[R2]` §7.2 M4 执行重训与 Gate-E。

---

## 4. 未解决问题及后续解决策略

| 优先级 | 问题 | 策略与实施步骤 | 时间节点 |
|---|---|---|---|
| 高 | M0 ⑤（真实 perturb smoke + P50/P95/RSS/显存基线）——环境已就绪 | 用已验证 tensor key + 词表在 `perturbgen` 环境执行：最小双路径 workload、资源基线采集；evidence 累积进 `outputs/perturbgen/` | 2026-08-24～08-25（`[R3]` §5 M0 窗口） |
| 高 | 合规 donor 队列缺失（M0⑥，Gate-0 硬阻塞） | 获取 raw counts、无版本 ENSG、cell type/state、明确 donor 标注的 normal/disease AnnData（≥3 可评估 donor）；不得把 sample/batch/replicate 当 donor（`[R2]` §4.6） | 2026-08-26～08-30（M1 窗口；外部数据依赖，无法代码侧推进） |
| 中 | DAVF 重训 + Gate-E | Gate-0 通过后按 `[R2]` §7.2 M4：`embedding_asset_path` 指向真实资产重训、冻结 ≥200 PTM→gene 基准、新旧 embedding 配对指标 + bootstrap CI | 2026-09-10～09-16 |
| 中 | Gate-4 真实资产 evidence / M5 | self-hosted GPU workflow 双路径真实 smoke（两独立 h5ad、峰值显存、P50/P95） | 2026-09-17～09-21 |
| 低 | Geneformer 主链物理删除（M7） | Gate-E + Gate-5 双过后删除 `geneformer_embedding.py` 主链、旧配置、专属测试；届时 `rg geneformer` 应只剩 §1.3 保留边界 | 2026-10-05～10-06 |
| 低 | mypy 既有 23 错误 / 共享环境 pip check 冲突 | 沿用 `[R3]` §5 并行收尾项策略（不扩大豁免、不在共享环境盲目降级）；本轮已确认改动文件 0 新增 | 与 M4 并行 |
| 低 | 90s daemon worker 超时资源监控 | 沿用 `[R3]` §5：真实部署出现高频黑洞再移进程边界 | 观察项 |

---

## 5. 可复现命令

```bash
# 环境搭建（幂等，含 evidence 采集）
bash scripts/setup_perturbgen_env.sh

# 不稳定网络下的持久化 wheel 下载（curl 断点续传 + sha256 校验，已实测 95/95）
# 先生成 pip dry-run 报告，再下载，最后离线安装：
#   pip install --dry-run --report /tmp/pip_report.json <锁定包列表>
bash scripts/download_perturbgen_wheels.sh /tmp/pip_report.json
pip install --no-index --find-links ~/perturbgen_wheels <锁定包列表>

# 仅采集 M0 环境 evidence（需在 perturbgen 环境中）
/home/scu/anaconda3/envs/perturbgen/bin/python scripts/collect_perturbgen_env_evidence.py

# GPU 实测：独立环境加载真实 checkpoint
/home/scu/anaconda3/envs/perturbgen/bin/python -c "
import torch
ckpt = torch.load('perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt', map_location='cuda', weights_only=False)
print(ckpt['state_dict']['transformer.token_embedding.weight'].shape)"

# 全量离线测试（本轮门禁口径）
unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest tests -q -p no:cacheprovider

# schema v2 注入链路真实资产验证
python -c "
from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule
m = DAVFInferenceModule(DAVFInferenceConfig(
    checkpoint_path='checkpoints/absent.pt',
    embedding_asset_path='outputs/perturbgen/embedding_asset_20260822'))
print(m.latent_davf.gene_embed_table.weight.shape)"

# 静态门禁
python -m compileall -q src scripts tests && ruff check src scripts tests
```

---

## 6. 最终判定

| 判定项 | 状态 |
|---|---|
| Python 3.11 独立环境（conda + commit/GPU/依赖锁定 + 复现脚本 + evidence JSON） | ✅ 完成 |
| 锁定依赖安装（95 wheels sha256 全验证 + 离线安装 + editable perturbgen） | ✅ 完成 |
| GPU 实测（cuda_available、真实 checkpoint cuda:0 加载、GPU matmul、CLI 四子命令） | ✅ 完成 |
| M0 token round-trip | ✅ 完成 |
| Geneformer 死语义删除（schema v2 迁移闸门） | ✅ 完成 |
| PerturbGen embedding 注入链路（含真实 18967×768 验证） | ✅ 完成 |
| checkpoint schema 显式版本化（写+读） | ✅ 完成 |
| finetune E2E 脚本资产参数（A3） | ✅ 完成 |
| 全量回归（2321 passed / exit 0）+ 静态门禁 | ✅ 完成 |
| M0 ⑤ 真实 perturb smoke + 资源基线 | ❌ 待 M0 窗口执行（需真实 tokenized 数据） |
| donor 合规队列（M0⑥） | ❌ 阻塞（外部数据） |
| DAVF 重训/Gate-E/Gate-4/Gate-5 | ❌ 未开始（按 `[R2]` §7.3 停止条件有意等待） |

本轮边界声明：不把「注入链路代码就绪 + 合成/静态验证」写成「DAVF 已迁移到 PerturbGen 底座」；runtime 默认链路未切换，DAVF 重训与科学验收（Gate-E/5）仍是硬门槛，与 `[R2]` §7.3 一致。

**文档版本**：v1.0（2026-08-23）
