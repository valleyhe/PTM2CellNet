# PTM2CellNet 项目综合分析报告

**报告日期**：2026-08-20
**分析基线**：本地 `main` @ `cea16f3`（领先 `origin/main` @ `c76fff8` 22 个提交，工作树干净）
**上一份权威报告**：`project_analysis_20260816.md`（v5.0，已作为历史快照归档至 `archive/20260820/reports/`）
**执行方式**：主代理负责版本控制/归档/验证门禁，3 个只读分析子代理并行执行任务 1/2/3（调用统计见 §6）

---

## 目录

- [摘要](#摘要)
- [1. 前置操作记录](#1-前置操作记录)
  - [1.1 版本控制操作](#11-版本控制操作)
  - [1.2 编译与验证](#12-编译与验证)
  - [1.3 文档与报告归档](#13-文档与报告归档)
- [2. 任务 1：未实现功能项识别与记录](#2-任务-1未实现功能项识别与记录)
- [3. 任务 2：未完全实现功能梳理](#3-任务-2未完全实现功能梳理)
- [4. 任务 3：技术债识别、分类与解决策略](#4-任务-3技术债识别分类与解决策略)
- [5. 结论与建议](#5-结论与建议)
- [6. 任务执行统计](#6-任务执行统计)
- [附录 A：验证命令与结果汇总](#附录-a验证命令与结果汇总)
- [附录 B：本轮修复的 lint 清单](#附录-b本轮修复的-lint-清单)

---

## 摘要

本轮综合处理结论如下：

1. **版本控制**：起点工作树干净、无待提交修改；本地 `main`（`cea16f3`）为远程 `origin/main`（`c76fff8`，fetch 确认无新提交）的超集（领先 22 / 落后 0），无需合并、无冲突。本轮在验证与归档完成后产生 1 个新提交（版本记录见 §1.1）。
2. **编译与测试验证**：`compileall` 通过；单元测试 **1957 passed / 6 skipped**（+241 例较 08-16 基线）；集成测试 **107 passed / 1 skipped**（650.92s）；`ruff` 发现 6 处新增 lint（全部在 `tests/`），本轮已修复并复测全绿；`mypy` 在当前依赖齐全环境实测 **25 errors / 12 files**，与 08-16 文档自述"0 errors"不符，登记为新增技术债 TD-N-06。
3. **归档**：识别出 2 份过时文件（`project_analysis_20260816.md` 综合报告、`.planning/v1.0-MILESTONE-AUDIT.md` 审计快照），归档至 `archive/20260820/`，附 MANIFEST（版本标签）与原路径重定向短页；`docs/` 下 7 份 08-04/08-08 报告经核已是既往批次的归档入口短页，无需重复处理；`docs/CURRENT_STATUS.md` 5 处过时内容已按实测刷新。
4. **任务 1（未实现项）**：对照正式需求规范（REQUIREMENTS.md v2.1/v2.2 全 Complete、ROADMAP.md 20 个 Phase 全 Complete），**不存在"已明确定义但完全未实现"的代码功能项**。剩余 2 项非代码缺口：GAP-1 真实数据科学验收流程（F-04，高优先级，核心影响）；GAP-2 replogle/scgenescope 受控数据导入（中优先级，边缘影响，且已满足 DATA-01"显式登记受控资产"契约）。
5. **任务 2（完成度）**：8 个模块完成度 **90.0%～98.0%**（整体加权约 95.3%），`src/` 全库仅 1 处 TODO、零 `raise NotImplementedError`。主要缺口集中在可选依赖功能（Mamba CUDA、ESM-2/ProtT5/scVI）、受控数据集与真实资产验收。
6. **任务 3（技术债）**：新增技术债 **7 项**（子代理 5 项 + 主代理验证门禁发现 2 项），其中 **0 严重 / 0 高 / 4 中 / 3 低**；同时核实 AGENTS.md 自述的 4 项存量技术债中 **3 项已实际闭环、1 项描述过时**（文档滞后，需更新 AGENTS.md 头部注记）。中级以上债均给出多方案解决策略与 0.5 天精度的时间估计。

---

## 1. 前置操作记录

### 1.1 版本控制操作

| 项目 | 值 | 证据 |
|---|---|---|
| 合并前本地 `main` | `cea16f3`（test: decouple GSE90546 probe-semantics guard…） | `git log --oneline -1` |
| 远程 `origin/main` | `c76fff8`（docs: record remote main publication），`git fetch` 后确认远程无新提交 | `git rev-parse origin/main` |
| 领先/落后关系 | **领先 22、落后 0**（本地为远程严格超集） | `git rev-list --left-right --count main...origin/main` → `22 0` |
| 工作树状态 | 干净，无待提交代码修改（前置"提交所有已完成修改"自然满足） | `git status` → `nothing to commit, working tree clean` |
| 合并操作 | **无需执行**：已位于 `main`，无待合并分支；`audit/20260809-*` 为历史审计分支，不在合并范围 | `git branch -a` |
| 冲突 | 无（无合并发生；本地包含远程全部历史） | 同上 |
| 本轮新提交 | **3 个提交**：`d703430`（style(tests)：6 处 ruff 修复）、`d0a3b0c`（归档批次 + 本报告，含 R100 纯 rename 保历史）、`a5ab6b2`（原路径重定向短页）。合并前版本 `cea16f3` → 合并后版本 `a5ab6b2` | §1.3、附录 B |

**本地领先远程的 22 个提交关键修改点**（按主题归类，`git log --oneline origin/main..main`）：

| 主题 | 提交 |
|---|---|
| 数据管线收口 | `32ded55` Norman/Adamson 导入 + OmniPath 激酶-底物；`4d02a87` manifest 契约缺口（GSE90546 目录、EPSD/scPerturb 登记、TD-H02 预计算器）；`aefd994` BioPlex/RegNetwork/STRING 规范图导入器（F-03）；`565e29f` F-03 尾巴（真实 STRING ENSP 映射 + RegNetwork 生物学分型）；`ed195da` scPerturb h5ad 解析器（F-05） |
| API 在线服务 | `a63e4dd` 跨尺度三端点 initialize/predict/batch_predict（F-01）；`9c87800` 阻塞推理卸载线程池；`a0cfb61` predict_variant 函数分解（TD-LONG-01/N12） |
| 训练正确性 | `bcb9cc9` 预训练 checkpoint 完整状态与精确恢复（N20） |
| 质量与文档 | `3da79c8` K02 fused CUDA 验收锁定；`6dc4472` PTM 窗口常量集中化；`33f6bde` UniProt 轮询指数退避；`bc5a74a` setup.py 单一依赖权威；`6ae2d4e` requirements-lock 入库；`4e5e234` 分析报告 v5.0；`85998f7` Geneformer 契约测试套件 + 部署指南；`cea16f3` GSE90546 测试解耦 |

### 1.2 编译与验证

全部命令于 2026-08-20 在当前环境实际执行（Python 3.12.13、PyTorch 2.4.1+cu118）：

| 门禁 | 命令 | 结果 |
|---|---|---|
| 字节码编译 | `python -m compileall -q src scripts` | **exit 0** |
| 单元测试 | `python -m pytest tests/unit -q` | **1957 passed, 6 skipped**（98.69s；复跑 110.79s 结果一致） |
| 集成测试 | `python -m pytest tests/integration -q` | **107 passed, 1 skipped**（650.92s） |
| Lint | `ruff check src scripts tests` | 首轮 **6 errors**（全在 `tests/`）→ 本轮修复后 **All checks passed**（附录 B） |
| 类型检查 | `mypy src` | **25 errors / 12 files**（详见 TD-N-06；`pyproject.toml:38-44` 对 architectures、predictions 两模块豁免部分错误码后仍余 25） |

**6 个单元测试 skip 的实测原因**（`-rs` 捕获）：2× `test_scvi_adapter.py:109/114`（scvi-tools 已安装的反向守卫）、1× `test_training_inference_consistency.py:129`（pooling 实现不支持整模型 pickle）、1× `:266`（ONNX 库缺失）、2× `test_window_constants.py:70`（序列外位置边界守卫）。**1 个集成测试 skip**：`test_data_contract_fixes.py` 三处数据快照守卫之一触发（`pytest.skip("scperturb/epsd/GSE90546 snapshots not on disk")`，`tests/integration/test_data_contract_fixes.py:42/91/120`）。

**结论：编译流程验证通过**（compileall 通过、全部测试无失败）；ruff 门禁经本轮修复恢复绿色；mypy 回归登记为技术债 TD-N-06，不阻塞运行时（测试全绿佐证）。

### 1.3 文档与报告归档

归档目录：`archive/20260820/`（`reports/` + `planning/` + `MANIFEST.md` + `README.md`）。

**判定标准**（沿用 `archive/20260816/MANIFEST.md` 规则）：过时 = 内容与当前代码实现或需求规范存在实质性差异；过期 = 生成超过 30 天或内容不能反映当前项目状态。

| 归档文件 | 原路径 | 版本标签 | 归档依据 |
|---|---|---|---|
| `reports/project_analysis_20260816.md` | 仓库根 | `4e5e234`（2026-08-17，`git mv` 保留历史） | 测试基线 1716/4 已变为 1957/6；发布后 5 个实质性提交改变其技术债/缺口结论；被本报告取代 |
| `planning/v1.0-MILESTONE-AUDIT.md` | `.planning/` | 未追踪（front matter 自记 `audited: 2026-04-05`，>30 天） | 快照性质，其后 5 轮系统性复核已改变其记录的全部 gap 状态 |

**判定为无需归档的边界项**（完整清单及依据见 `archive/20260820/MANIFEST.md` §3）：`docs/` 下 7 份 08-04/08-08 报告（正文已在 `archive/20260808/`、`archive/20260816/`，现仅剩重定向入口短页）；3 份 `project_repair_report_20260816/17*.md`（修复轮审计记录，未被取代）；`.planning/MILESTONES.md`（累积记录）；`data/README.md`（数据源描述与 `data/` 现有布局一致）；`CLAUDE.md`/`codex_mcp_prompt.md`（工具链配置/任务输入）。

**原路径重定向**：`./project_analysis_20260816.md` 现为指向 `archive/20260820/reports/` 与本报告的入口短页（沿用既往惯例，活动入口不删除）。

**`docs/CURRENT_STATUS.md` 同步刷新（5 处，均有实测依据）**：更新日期与权威报告指向；测试基线 1716/4 → 1957/6（含 `-rs` 实测 skip 原因）；静态质量行改记 ruff 修复与 mypy 25 errors 实况；剩余需求差距行改记 replogle/scgenescope 2 项 controlled（scperturb 已 implemented，`data/manifests/datasets.yaml` 实测）与跨尺度 API/预计算器/图导入的闭环状态。

---

## 2. 任务 1：未实现功能项识别与记录

> 由 Explore 子代理执行（只读），主代理对其关键论断做了独立复核。

### 2.1 识别方法

1. 以**正式需求规范**为对照基准：`.planning/REQUIREMENTS.md`（v2.1/v2.2 需求条目全部标注 Complete）、`.planning/ROADMAP.md`（20 个 Phase 全部 Complete）、`AGENTS.md` §0 实际代码映射。
2. 排除已裁剪范围：2026-07-05 需求裁剪决定（实时质谱流 V2-02、自定义 PTM 数据库 V2-03、GUI、API key 扩展）不列为未实现（`docs/CURRENT_STATUS.md:21` 范围裁剪行；测试中对应 DeprecationWarning 佐证：`tests/unit/test_roadmap.py:100/122`）。
3. 逐项 grep 验证：API 端点逐一比对 `src/api/routes/`（predictions/initialize/model_info/cross_scale/monitoring）；75 个模块文件存在性检查；历史报告登记的"未实现"项逐个用当前代码状态核销。

### 2.2 未实现项清单

**结论：不存在"已明确定义但完全未实现"的代码功能项或缺失 API 接口。** 剩余 2 项非代码功能缺口：

| 编号 | 功能/接口名称 | 需求出处 | 优先级 | 影响范围 | 证据 |
|---|---|---|---|---|---|
| GAP-1 | 真实数据科学验收流程（固定快照/指标阈值/验收报告，F-04） | `docs/CURRENT_STATUS.md:19`（剩余需求差距）；`project_analysis_20260816.md` §6.2「F-04 ❌ 不满足」（已归档）；README「真实资产验收」门禁说明 | **高** | **核心**（科学有效性验证） | `tests/real_assets/test_real_cptac_validation.py:43` `pytestmark = skipif(PTM2CELLNET_RUN_REAL_ASSET_TESTS)`，全部真实资产测试按门禁 skip；仓库无固定阈值/快照/验收报告文件 |
| GAP-2 | replogle / scgenescope 受控数据导入 | REQUIREMENTS.md DATA-01（受控资产显式登记即满足契约）；`data/manifests/datasets.yaml`（两数据集 `loader: null, status: controlled`，主代理实测复核） | 中 | 边缘（数据侧） | `scripts/import_*.py` 脚本集中无二者解析器；**但 DATA-01 契约允许 controlled 状态显式登记，严格说不构成工程违约** |

### 2.3 未闭环业务流程（GAP-1 主流程）

```
原始数据 → manifest 登记 → 导入器(bioplex/regnetwork/string/kinase/norman/scperturb) → canonical 边表/表达 → 跨尺度 NPZ
      :replogle/scgenescope: ──✗ loader:null（受控，仅登记，无导入）♯GAP-2
        ↓
train_cross_scale.py → CrossScalePTM2CellNet → /cross-scale/* API（已闭环，a63e4dd）
        ↓
  ✗✗ 真实科学验收：tests/real_assets 门禁内 skip、无指标阈值/快照/验收报告 ♯GAP-1 ✗✗
```

### 2.4 已核销的"曾称未实现"项（防回归清单）

以下项在 08-16 及更早报告中被登记为未实现/待办，本轮逐一核销（列出关键证据）：

| 项 | 核销证据 |
|---|---|
| 跨尺度 API 三端点 | `src/api/routes/cross_scale.py:239/686/731`；路由接线 `src/api/routes/__init__.py:20`（主代理复核）；提交 `a63e4dd` |
| TD-H02 embedding 预计算器 | `scripts/prepare_cross_scale_embeddings.py` 存在（主代理复核）+ 专门测试 `tests/unit/scripts/test_prepare_cross_scale_embeddings.py`；提交 `4d02a87` |
| 信号图原始格式导入 | `scripts/import_ppi_graphs.py`（Bioplex/RegNetwork/STRING）；提交 `aefd994` + `565e29f` |
| scperturb h5ad 解析 | `scripts/import_scperturb.py`；manifest `status: implemented`（主代理实测）；提交 `ed195da` |
| GSE90546 解析 | `scripts/import_norman_adamson.py:474 parse_gse90546`，CLI `--parse-gse90546`（`:766`）；配套测试解耦 `cea16f3` |
| Mamba fused CUDA 路径 | `src/models/mamba_encoder.py:237 _ssm_step_fused`；验收锁定 `3da79c8` |
| 跨尺度 API 文档 | `docs/guides/deployment.md:141-157`（`85998f7`） |

---

## 3. 任务 2：未完全实现功能梳理

> 由 general-purpose 子代理执行。完成度为综合文件规模、契约覆盖、测试覆盖与文档证据的估计值（±3%）。

### 3.1 模块完成度总表

| 模块 | 完成度 | 判定依据摘要（关键契约落点） | 缺失组件/依赖 |
|---|---|---|---|
| `src/data/` | 97.5% | DataLoader（mixin 组装 `loaders/`）、DataPreprocessor 三契约方法（`clean_sequences:96`/`normalize_ptm_labels:139`/`split_dataset:212`）、FeatureExtractor（785 行，`extract_*:315/589/711`）、PTMDataset:88、DAVFSiteAugmenter、homology_splitter、data_manifest/data_contract | 蓝图类名 `PTMDataModule` 不存在（被 4 个具体 DataModule 取代，命名偏差） |
| `src/models/` | 95.0% | 全编码器（CNN:43/LSTM:113/GRU:146/Transformer:61/Mamba:373）、PTMEmbedding:12/PTMAttention:176、PTM2CellNetBase:83/PTM2CellNet:677/Large:655、DAVF 全链路、ensemble、variant_effect、signaling_network、geneformer | Mamba 无 CUDA 时串行（`mamba_encoder.py:168`）；ESM-2/ProtT5/scVI 为可选依赖 |
| `src/training/` | 95.5% | Trainer:31、Focal:14/MultiTask:221、configure_optimizer:20、ModelCheckpoint:126/EarlyStopping:339、artifacts.py、self_supervised（782 行） | `self_supervised.py` 单测覆盖 31.62%（`docs/TEST_COVERAGE.md:57`）——测试缺口而非实现缺口 |
| `src/evaluation/` | 98.0% | `calculate_accuracy:41`/`f1:68`/`auc:79`、独立 `evaluate`/`cross_validate`（`evaluators.py:236`）、explainers、visualization | 无 |
| `src/utils/` | 98.0% | SafeUnpickler + `save_model:265`/`load_model:278`、`setup_logger:67`、Config:33、helpers、checkpoint_utils | 无 |
| `src/api/` | 96.0% | 工厂 + 限速/Prometheus/请求体限制/CORS/API-key 中间件（`app.py:97-176`）、全部标准端点 + 跨尺度三端点、`PTMSite.gene_symbol:29` + `use_davf:68` | 无（跨尺度路由已接入，`a63e4dd`） |
| `src/analysis/` + `src/integration/` | 92.5% | variant_workflow/parser/gene_mapper/pathway_integration（sspa 可选）+ pdc_client；GenKI 全套；ptm_virtual_perturbation + contracts | KEGG/Reactome 完整集成为需求显式 out-of-scope（`REQUIREMENTS.md:99`）；GenKI 离线阵列后端读 `mock_*` 文件（`reference_data.py:121`） |
| `scripts/` | 90.0% | 41 个 CLI 齐全（含 8 月新增 import_norman_adamson/kinase_substrate/ppi_graphs/scperturb） | `baselines/pmads_ridge.py` 为全仓库唯一 TODO 处 |

**代码卫生实测**：`src/` 全库仅 1 处 TODO（`baselines/pmads_ridge.py`）、零 `raise NotImplementedError`（唯一在 `encoders.py:19` 抽象基类，属设计）、所有 `pass` 均为空 mixin 类体/异常吞噬/`==` 操作符返回 `NotImplemented`（`external_tools/base.py:206`，Python 惯用法）等合规用法。

### 3.2 与需求规范的差异明细

| 需求/规范原文 | 实际实现 | 差异性质 |
|---|---|---|
| AGENTS 蓝图 `class PTMDataModule`（§数据模块接口契约） | `PTMLightningDataModule`（`lightning_datamodule.py:21`）、`PTMPlainDataModule`（`datasets.py:402`）等 4 个 | 类名契约不符，功能全覆盖，属命名偏差 |
| CUDA Mamba 应可用（环境齐备时） | 无 CUDA/未装 mamba_ssm 时走串行实现（`mamba_encoder.py:168` `_HAS_MAMBA_SSM and x.is_cuda`） | 性能缺陷（非正确性），AGENTS.md 自列技术债 |
| SignalingNetworkMapper KEGG/Reactome 集成 | 失败时降级为内置通路集（9 条，`signaling_network.py:271-283`） | 被 `REQUIREMENTS.md:99` 与 CONF-03 显式豁免（sanctioned 降级） |
| CURRENT_STATUS.md:19 旧文"跨尺度未接入 API" | `routes/__init__.py:20` 已 include cross_scale_router（`a63e4dd`） | 文档滞后于代码（本轮已刷新该行） |
| "禁止隐式随机 fallback"（AGENTS 工程规范） | `geneformer_embedding.py:244-255` 随机嵌入 fallback（可被 strict_assets 关闭）+ cross_scale 响应 `fallback_flags` 显式上报 | 已显式化、风险受控；非严格模式仍可能产出随机嵌入预测 |

### 3.3 三分类清单（含判断标准）

**一、部分实现但可用**——判断标准：核心路径可运行且有替代路径，缺口有明确文档登记。
- YAML 数据清单：9 个 required 数据集余 replogle/scgenescope 2 项 controlled（`data/manifests/datasets.yaml`，主代理实测）。
- 真实资产验收未启用（`tests/real_assets/` 按门禁 skip，即 GAP-1）："全绿"≠科学验收。

**二、实现但有缺陷**——判断标准：代码可运行且测试通过，但存在已登记的性能/依赖/测试缺口。
- Mamba 串行实现（`mamba_encoder.py:168`，无 CUDA 时性能损失）。
- `pip check` 两个可选依赖冲突（scgpt 0.2.4 与 scvi-tools 1.4.3 并存，`requirements-lock.txt:207/214`，已登记 F-10）。
- `self_supervised.py` 单测覆盖 31.62% 偏低（`docs/TEST_COVERAGE.md:57`），回归风险。
- mypy 25 errors（本轮新发现，见 TD-N-06）。

**三、实现但不符合规范**——判断标准：行为语义等价但类名/签名/存放位置与契约文字不一致。
- `PTMDataModule` 蓝图类名未提供（见 §3.2）。
- Preprocessor/FeatureExtractor 契约签名：`clean_sequences` 增加 `sequence_col` 参数、`split_dataset` 签名扩展——向前兼容但非严格一致。
- 根 `signaling_network.py` 为 re-export shim（`REQUIREMENTS.md:25` CODE-03 已豁免）。

---

## 4. 任务 3：技术债识别、分类与解决策略

> 由 general-purpose 子代理执行（5 项新增），主代理补充验证门禁类 2 项（TD-N-06/07）。已先读 6 份现有报告建立"已提及"基线，仅报告新增项。

### 4.1 分级标准

| 级别 | 标准 |
|---|---|
| 严重 | 影响正确性或安全性，直接引发错误数据/错误行为，或阻塞主要功能 |
| 高 | 显著损害可维护性或性能，未来大概率引发 bug |
| 中 | 明显增加维护成本，但当前不产生错误 |
| 低 | 风格/整洁层面问题 |

### 4.2 存量技术债核销核实（AGENTS.md 自述 4 项）

AGENTS.md 头部"仍存技术债（非阻断）"注记已过时，实测核实：

| AGENTS.md 自述 | 实测现状 |
|---|---|
| "10 个文件 mypy: ignore-errors" | `src/` 内 `ignore-errors` 与顶部 `type: ignore` 均 0 命中；实为 `pyproject.toml:38-44` 两个 `disable_error_code` override（architectures、predictions）——"10 个文件"描述失实 |
| "Mamba SSM 串行实现" | 主路径 fused/parallel 已实现（`3da79c8` 验收锁定）；串行仅为无 CUDA 回退，属设计而非债 |
| "长序列窗口逻辑重复" | `long_sequence.py` 已委托 `LongSequenceHandler`，重复已消除 |
| "`estimate_max_batch_size` 硬编码层数" | 已由 `_infer_activation_depth`（`model_utils.py:215-260`）从 `named_modules` 实际探测 + config 覆盖修复，仅注释残留历史值 |

**衍生文档债**：AGENTS.md 头部技术债注记需更新（建议随下一轮提交修订）。

### 4.3 新增技术债清单

| 编号 | 类别（SonarQube 参照） | 严重度 | 问题 | 证据 |
|---|---|---|---|---|
| TD-N-01 | Dead code / unused | 中 | `src/data/loaders/legacy_loaders.py` 为已取消功能（V2-02/V2-03）的僵尸模块，全仓零代码调用 | `grep -rn "legacy_loaders\|LegacyLoader" src/` 仅命中 `src/models/roadmap.py:10`（docstring）；文件头部自述 "Legacy loaders for cancelled roadmap features" |
| TD-N-02 | Test coverage | 中 | `src/utils/lazy_import.py` 零测试；它是 evaluation/analysis 两包 `__init__.py` 的可选依赖懒加载垫片 | `grep -rln "lazy_import" tests/` 0 命中；`src/evaluation/__init__.py`、`src/analysis/__init__.py` 使用 `LazyImport` |
| TD-N-03 | Documentation | 中 | 在线端点 `/predict/variant`（自定义迁移策略 + 置信度，刚经 `a0cfb61` 重构）在部署文档中完全缺失 | `grep -ni "variant" docs/guides/deployment.md` 0 命中；端点在 `predictions.py:752` 且有专门测试 |
| TD-N-04 | Duplicated code | 低 | `encoders.py` 三个 Pooled 变体 `__init__+forward` 结构同构（pooling 选择/维度断言/掩码传递） | AST 统计三类方法行数 25/19/17 + 6/6/8；`src/models/pooling.py` 已有 `create_pooling_layer` 工厂可收敛 |
| TD-N-05 | Performance | 低 | 14 处 `df.iterrows()` | `grep -rn "\.iterrows()" src/` 14 命中；抽查均在数据集构建/校验期（`multitask_dataset.py:105`、`data_contract.py:190`），非训练热路径，仅超大型输入退化 |
| TD-N-06 | Reliability / type-safety | 中 | **mypy 25 errors / 12 files**（本轮验证发现）：`Tensor \| Module` 联合类型误用（`long_sequence.py:191-243`、`peft_config.py:133-287`、`ensemble.py:112/149` 等）、`str \| None` 未收窄（`cross_scale.py:466/706/714`）、`no-any-return`（`pmads_ridge.py:160/378`、`features.py:355`）；且 08-16 文档自述"0 errors"与实测不符（依赖装齐后错误暴露） | `mypy src --show-error-codes` 全量输出（§1.2）；`pyproject.toml:28-44` |
| TD-N-07 | Code hygiene | 低 | 仓库根存在 6 个 0 字节垃圾文件（`0.7692`、`=0.8.0`、`=1.0.0`、`=2.5.0`、`=3.0.0`、`EOF`，疑似 pip 命令参数误重定向产物）；另有会话产物（`omp-session-*.html`、`x.pt`、`*.zip`）散落根目录 | `file` 验证均为 empty；`git check-ignore` 确认被 `.gitignore:6`（`/*` 白名单）忽略，不入库但污染工作区 |

**未发现新的"严重/高"级技术债**——多轮修复报告（2026-08-16/17/18）已将高债基本清空，存量高优先项（F-04 验收、F-06/F-07 数据依赖）为需求/数据侧缺口而非代码债。

### 4.4 中级技术债解决策略

#### TD-N-01 legacy_loaders 死代码（中）

- **影响**：为已删除的 roadmap 需求保留约 250 行加载实现与列 schema，徒增维护面与误用面。
- **方案 A（推荐）**：删除模块，`roadmap.py` docstring 改为"已移除"标记。优点：彻底；缺点：外部脚本若曾 import 会报错——已全仓 grep 确认零引用。
- **方案 B**：迁入 `docs/archive/` 作历史样本。优点：可追溯；缺点：归档代码无人维护、易腐烂。
- **方案 C**：标记 `@deprecated` 延迟删除。优点：渐进；缺点：无硬 deadline。
- **实施**：0.5 工作日（删除 + docstring 修正 + 聚合入口确认）。资源：无特殊要求。风险：极低。

#### TD-N-02 lazy_import 补测（中）

- **影响**：可选依赖垫片的失败语义若退化，两个包的导入行为会静默变化，现有测试只测成功路径。
- **方案 A（推荐）**：新增 `tests/unit/test_lazy_import.py` 契约测试——成功加载、ImportError 后 `get()` 重新抛出、`attr_name` 缺失抛 AttributeError、两次调用缓存命中不二次 import。
- **方案 B**：改用 `importlib.util.find_spec` 前置探测 + 显式异常。改动大、收益低，不推荐。
- **实施**：0.5 工作日。风险：无。

#### TD-N-03 deployment 文档补录 /predict/variant（中）

- **影响**：运维与 API 消费者无从得知该端点的参数与错误语义（400/503/504），属应有公开契约。
- **方案 A（推荐）**：`docs/guides/deployment.md` 增加"变体预测端点"小节（请求/响应/错误码 + 与 `/predict` 差异）。
- **方案 B**：仅依赖 OpenAPI `/docs` 自动生成。缺点：离线不可见、与现有文档风格不一致。
- **实施**：0.5 工作日。风险：需与 `a0cfb61` 拆分后的行为核对（易把 503 误写为 500）。

#### TD-N-06 mypy 25 errors 清偿（中）

- **影响**：`Tensor | Module` 联合误标注掩盖真实类型约束，重构时易引入静默类型错误；"0 errors"失实陈述削弱质量门禁可信度。
- **方案 A（推荐，分两批）**：第一批 0.5 天修纯注解问题（`str | None` 收窄加 `assert x is not None` 或显式分支：`cross_scale.py:466/706/714`、`predictions.py:666/770`；`no-any-return` 加 `typing.cast` 或精确返回类型：3 处）；第二批 1.0 天修 `Tensor | Module` 联合（`long_sequence.py`/`peft_config.py`/`ensemble.py`/`lightning_module.py:163`——根因是 forward 引用注解不精确，需拆分类型别名或改用 `Module` 基类并收窄）。修后 `mypy src` 入 CI 门禁。
- **方案 B**：在 `pyproject.toml` 追加 `disable_error_code` 豁免涉及文件。优点：立即归零；缺点：掩盖真实约束，债转隐式。
- **方案 C**：维持现状仅记录。缺点：与"0 errors"自述长期背离。
- **实施**：合计 1.5 工作日。资源：需熟悉 PyTorch 类型生态的工程师。风险：`cast` 滥用会引入假阴性，需配合单测抽查。

**低级债（TD-N-04/05/07）处理建议**：TD-N-04 用 `create_pooling_layer` 工厂收敛三个 Pooled 变体（1 天，可并入下次 encoder 改动）；TD-N-05 保持观察（非热路径）；TD-N-07 直接清理根目录垃圾文件与陈旧会话产物（0.5 天，注意先确认 `*.zip`/`x.pt` 无归属再删）。

---

## 5. 结论与建议

1. **需求符合度**：正式需求规范定义的代码功能已全部实现（含 API 接口层），项目工程完成度整体约 95.3%（8 模块 90.0%–98.0%）。仅剩 2 项非代码缺口：**GAP-1 真实数据科学验收（F-04）**为当前最高优先事项——建议为其定义固定数据快照、指标阈值与验收报告模板并接入 `tests/real_assets/` 门禁（预计 2–3 工作日，需真实数据授权）；GAP-2 维持 controlled 登记即满足契约，无行动必要。
2. **质量门禁实况**：编译/单测/集成/lint 全绿（本轮修复 6 处 tests/ lint）；**mypy 25 errors 是本轮最重要的新发现**，建议按 TD-N-06 方案 A 分两批清偿后将 mypy 纳入固定门禁，避免"文档 0 errors"与实测长期背离。
3. **文档治理**：本轮归档 2 份过时文件并刷新 CURRENT_STATUS 5 处滞后内容；**AGENTS.md 头部技术债注记已过时（4 项中 3 项已闭环）**，建议下一轮提交时同步修订，避免后续代理/会话按过时债清单决策。
4. **技术债总量**：新增 7 项（0 严重/0 高/4 中/3 低），中级 4 项合计约 3.0 工作日可全部清偿（TD-N-01 0.5 + TD-N-02 0.5 + TD-N-03 0.5 + TD-N-06 1.5）。
5. **版本控制**：本地领先远程 22 个提交（5 大主题：数据管线收口、API 在线服务、训练正确性、质量与文档、验收锁定），建议择机 `git push origin main` 发布（本次任务范围限定本地操作，未推送）。

---

## 6. 任务执行统计

### 6.1 子智能体调用统计

| 智能体名称 | 类型 | 调用次数 | 主要执行任务 | 平均执行时长 | 工具调用数 |
|---|---|---|---|---|---|
| 任务 1 分析子代理 | Explore（只读） | 1 | 未实现功能项识别：需求对照、75 文件存在性检查、API 端点逐一比对、历史"未实现"项核销 | 395.4 秒（≈6.6 分钟） | 40 |
| 任务 2 分析子代理 | general-purpose | 1 | 模块完成度评估：契约落点核对、TODO/NotImplemented 全库扫描、三分类判定 | 282.3 秒（≈4.7 分钟） | 26 |
| 任务 3 分析子代理 | general-purpose | 1 | 技术债识别：6 份现有报告去重基线、AST/grep 实证、存量债核销核实、解决策略制定 | 558.2 秒（≈9.3 分钟） | 42 |
| **合计** | — | **3 次** | — | **412.0 秒/次（总计 1235.9 秒 ≈ 20.6 分钟，三代理并行墙钟 ≈ 9.3 分钟）** | 108 |

主代理（ZCode 会话本体，不计入子智能体统计）：版本控制操作、编译/测试/lint/mypy 五项验证、归档落地（MANIFEST/README/重定向短页）、CURRENT_STATUS 刷新、6 处 lint 修复、三份子代理结论的独立抽查复核、报告撰写与提交。

### 6.2 简要分析

三个子代理并行执行，墙钟耗时约 9.3 分钟（受最慢的任务 3 制约），串行等效时长 20.6 分钟，并行收益约 2.2 倍。任务 3 耗时最长（558 秒/42 次工具调用），符合其工作量特征——需先读 6 份既有报告建立去重基线再做增量识别；任务 2 最快（282 秒），因为模块完成度可大量复用 AGENTS.md §0 映射与 TEST_COVERAGE 自述，仅做抽样实证。主代理对子代理关键论断做了抽查复核（scperturb manifest 状态、跨尺度路由接线、脚本存在性、skip 原因实测），全部证实，未发现幻觉结论。

---

## 附录 A：验证命令与结果汇总

```text
$ git fetch origin && git rev-list --left-right --count main...origin/main
22 0                                     # 本地为远程超集，无需合并

$ python -m compileall -q src scripts; echo $?
0                                        # 编译通过

$ python -m pytest tests/unit -q
1957 passed, 6 skipped in 98.69s         # 复跑 110.79s 一致

$ python -m pytest tests/integration -q
107 passed, 1 skipped in 650.92s

$ ruff check src scripts tests
Found 6 errors → 修复后: All checks passed!（附录 B）

$ mypy src
Found 25 errors in 12 files (checked 134 source files)   # TD-N-06
```

## 附录 B：本轮修复的 lint 清单

| 文件:行 | 规则 | 修复方式 |
|---|---|---|
| `tests/integration/test_cross_scale_api.py:9` | F401 unused `json` | `ruff --fix` 删除 |
| `tests/integration/test_data_contract_fixes.py:15` | F401 unused `json` | `ruff --fix` 删除 |
| `tests/unit/scripts/test_import_ppi_graphs.py:9` | F401 unused `Path` | `ruff --fix` 删除 |
| `tests/unit/test_mamba_encoder.py:294` | F401 unused `einsum` | `ruff --fix` 删除 |
| `tests/unit/test_gene_mapper.py:86` | B905 `zip()` 无 `strict=` | 手动加 `strict=True`（同列表 `[:-1]`/`[1:]` 切片长度恒相等，语义不变） |
| `tests/unit/models/test_geneformer_embedding.py:119` | B018 无效表达式 | 手动改为 `_ = loader.embeddings`（property 触发意图不变，仍在 `pytest.raises` 内） |

修复后验证：`ruff check src scripts tests` 全绿；受影响 4 个测试文件 56 例全部通过（14.61s）。

---

**报告版本**：v1.0（2026-08-20）
**下一份权威报告接替本报告时**，请将本文件归档至 `archive/YYYYMMDD/reports/` 并在原路径留重定向短页（沿用既有惯例）。
