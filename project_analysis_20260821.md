# PTM2CellNet 项目综合分析报告（2026-08-21）

**报告日期**：2026-08-21
**分析基线**：本地 `main` @ `919f4ed`（领先 `origin/main` @ `c76fff8` 26 个提交，工作树干净）
**上一份权威报告**：`project_analysis_20260820.md`（已作为历史快照归档至 `archive/20260821/reports/`，原路径留重定向短页）
**执行方式**：主代理直接执行全部前置操作、验证门禁与三项系统性复核任务；本轮未调用任何子智能体（统计见 §6）

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

---

## 摘要

本轮综合处理结论如下：

1. **版本控制**：起点工作树干净、无待提交修改；本地 `main`（`919f4ed`）为远程 `origin/main`（`c76fff8`，fetch 确认无新提交）的严格超集（领先 26 / 落后 0），无需合并、无冲突。编译验证通过（`compileall` exit 0）、单元测试 **1957 passed / 6 skipped** 复现、集成测试 **107 passed / 1 skipped**（740.19s）复现、`ruff` 全绿。
2. **mypy 结论修正（本轮最重要发现）**：0820 报告"25 errors / 12 files"实为 **mypy 版本差异**所致——同一代码库在 mypy 1.20.0 下报 25 errors / 12 files，在 mypy 2.1.0 下仅报 **5 errors / 2 files**。两个版本均已实测复核并定位根因（新版对 `Tensor | Module` 联合类型与 `str | None` 收窄的推断改进）。TD-N-06 的清偿范围据此修正。
3. **新增技术债 TD-N-08（严重度：高）**：首次使用 `pip-audit` 对 `requirements-lock.txt` 全量审计，发现 **64 个已知漏洞条目，涉及 12 个包**——pillow 12.1.1（26 条）、aiohttp 3.13.5（14 条）、transformers 4.57.6（3 条）、pytorch-lightning 1.9.5（3 条）等；其中 pytorch-lightning 1.9.5 与 lightning 2.6.5 双包共存于锁定清单。
4. **归档**：`project_analysis_20260820.md` 因 mypy 结论被实测推翻 + 漏洞审计缺位归档至 `archive/20260821/reports/`（`git mv` 保历史）；同步刷新 `docs/CURRENT_STATUS.md`、AGENTS.md 过时技术债注记、3 份 docs 入口短页的权威报告指向。
5. **任务 1（未实现项）**：对照 `.planning/REQUIREMENTS.md` v2.1/v2.2（24/24 条目 Complete）与 `.planning/ROADMAP.md`（18 个 Phase 全 Complete），**不存在已明确定义但完全未实现的代码功能项或缺失 API 接口**。剩余 2 项非代码缺口维持上轮判定：GAP-1 真实数据科学验收（F-04，高优先级）、GAP-2 replogle/scgenescope 受控数据导入（中优先级，契约允许 controlled 登记）。
6. **任务 2（完成度）**：10 个模块完成度 **90.0%～98.0%**（整体加权约 95.4%）；`src/` 全库 TODO/FIXME 计数为 **0**（上轮报告的 pmads_ridge 唯一 TODO 已消失），零 `raise NotImplementedError`（唯一一处为抽象基类设计）。
7. **任务 3（技术债）**：存量 7 项中 TD-N-06 范围修正、TD-N-01/02/03 维持、TD-N-07 部分清理评估完成；新增 TD-N-08（依赖漏洞，高）与 TD-N-09（coverage 与 torch 收集期冲突，低）。当前活跃技术债合计 **0 严重 / 1 高 / 4 中 / 4 低**。

---

## 1. 前置操作记录

### 1.1 版本控制操作

| 项目 | 值 | 证据 |
|---|---|---|
| 合并前本地 `main` | `919f4ed`（docs: record batch commit hashes in 20260820 report and MANIFEST） | `git log --oneline -1` |
| 远程 `origin/main` | `c76fff8`（docs: record remote main publication），`git fetch` 后确认远程无新提交 | `git rev-list --left-right --count main...origin/main` → `26 0` |
| 工作树状态 | 干净，无待提交代码修改（前置"提交所有已完成修改"自然满足） | `git status` → `nothing to commit, working tree clean` |
| 合并操作 | **无需执行**：已位于 `main`，无待合并分支；`audit/20260809-*` 为历史审计分支，不在合并范围 | `git branch -a -v` |
| 冲突 | 无（无合并发生；本地包含远程全部历史） | 同上 |
| 本轮新提交 | 见 §1.3 与文末提交记录；合并前版本 `919f4ed` → 归档后版本见 git log | §1.3 |

**本地领先远程的 26 个提交关键修改点**（`git log --oneline origin/main..main`，按主题归类）：

| 主题 | 提交 |
|---|---|
| 数据管线收口 | `32ded55` Norman/Adamson 导入 + OmniPath 激酶-底物；`4d02a87` manifest 契约缺口；`aefd994` BioPlex/RegNetwork/STRING 规范图导入器（F-03）；`565e29f` F-03 尾巴（真实 STRING ENSP 映射）；`ed195da` scPerturb h5ad 解析器（F-05） |
| API 在线服务 | `a63e4dd` 跨尺度三端点（F-01）；`9c87800` 阻塞推理卸载线程池；`a0cfb61` predict_variant 函数分解 |
| 训练正确性 | `bcb9cc9` 预训练 checkpoint 完整状态与精确恢复（N20） |
| 质量与文档 | `3da79c8` K02 fused CUDA 验收锁定；`6dc4472` PTM 窗口常量集中化；`33f6bde` UniProt 轮询指数退避；`bc5a74a` setup.py 单一依赖权威；`6ae2d4e` requirements-lock 入库；`4e5e234` 分析报告 v5.0；`85998f7` Geneformer 契约测试套件；`cea16f3` GSE90546 测试解耦；`d703430` 6 处 tests lint 修复；`d0a3b0c`/`a5ab6b2`/`919f4ed` 20260820 归档批次与本轮文档修订 |

### 1.2 编译与验证

全部命令于 2026-08-21 在当前环境实际执行（Python 3.12.13、PyTorch 2.4.1+cu118、CUDA 可用）：

| 门禁 | 命令 | 结果 |
|---|---|---|
| 字节码编译 | `python -m compileall -q src scripts` | **exit 0** |
| 单元测试 | `python -m pytest tests/unit -q` | **1957 passed, 6 skipped**（115.28s，与 0820 基线一致） |
| 集成测试 | `python -m pytest tests/integration -q` | **107 passed, 1 skipped**（740.19s，与 0820 基线 107/1 一致） |
| Lint | `ruff check src scripts tests` | **All checks passed** |
| 类型检查（mypy 2.1.0，SSH_unit env 默认） | `python -m mypy src` | **5 errors / 2 files**（明细见 §4.3 TD-N-06） |
| 类型检查（mypy 1.20.0，base env） | `/home/scu/anaconda3/bin/python -m mypy src` | **25 errors / 12 files**（与 0820 报告口径一致） |
| 依赖一致性 | `pip check` | 4 条已知冲突（datasketch 缺失、numpy<2 约束 vs 实装 2.4.3、scgpt vs scvi-tools、torchaudio 版本），均为登记在案的可选依赖取舍（F-10） |
| 安全审计 | `pip-audit -r requirements-lock.txt --disable-pip --no-deps` | **64 个漏洞条目 / 12 包**（新增 TD-N-08，详见 §4.3） |

**mypy 版本差异的根因定位**（本轮独立复核，两环境均清缓存后复测）：

```text
SSH_unit env: mypy 2.1.0  → Found 5 errors in 2 files (checked 134 source files)
base env:     mypy 1.20.0 → Found 25 errors in 12 files (checked 134 source files)
```

- 两份 `pyproject.toml` 配置完全相同（`bc5a74a` 后未再改动，`git diff HEAD -- pyproject.toml` 为空）。
- mypy 1.20 报错的 12 个文件分布：`peft_config.py`(5)、`long_sequence.py`(4)、`cross_scale.py`(3)、`predictions.py`(2)、`pmads_ridge.py`(2)、`ensemble.py`(2)、`cross_scale_trainer.py`(2) 及 features/cross_scale/encoders/model_utils/lightning_module 各 1。
- mypy 2.1 仅剩 `predictions.py`(2) 与 `cross_scale.py`(3)，全部为 `str | None` 未收窄 / `union-attr` / `var-annotated` 类问题；1.20 报告的 `Tensor | Module` 联合类型误用（`lightning_module.py:163` 等）在新版已被更精确的推断吸收。
- **结论**：0820 报告的"25 errors"数字本身可复现（用 mypy 1.20），但其未注明版本号导致跨会话不可比。CI（`.github/workflows/full-test.yml:57-60`）安装最新 mypy，故 CI 口径应同时跟踪两个数字；TD-N-06 清偿策略按 1.20 口径（超集）制定。

### 1.3 文档与报告归档

归档目录：`archive/20260821/`（`reports/` + `MANIFEST.md` + `README.md`）。

**判定标准**（沿用既往批次规则）：过时 = 内容与当前代码实现或需求规范存在实质性差异；过期 = 生成超过 30 天或内容不能反映当前项目状态。

| 归档文件 | 原路径 | 版本标签 | 归档依据 |
|---|---|---|---|
| `reports/project_analysis_20260820.md` | 仓库根 | `919f4ed`（2026-08-21 00:25，`git mv` 保留完整重命名历史） | 其 mypy "25 errors" 结论未注明版本且被双版本实测推翻口径；pip-audit 全量漏洞审计缺位；技术债清单被本报告取代 |

**判定为无需归档的边界项**（完整依据见 `archive/20260821/MANIFEST.md` §3）：`docs/` 下 8 份入口短页（正文已在既往批次归档，入口仍有效）；`.planning/{REQUIREMENTS,ROADMAP,PROJECT,STATE,MILESTONES}.md`（现行需求规范与 roadmap，是任务 1 对照基准）；`docs/TEST_COVERAGE.md`（现行覆盖率门禁依据）；`docs/guides/*.md` 6 份指南（与当前 CLI/API 一致）。

**原路径重定向**：`./project_analysis_20260820.md` 现为指向 `archive/20260821/reports/` 与本报告的入口短页（沿用既有惯例）。

**过时内容同步刷新（均有实测依据）**：

1. `docs/CURRENT_STATUS.md`：更新日期与权威报告指向；静态质量行改记 mypy 双版本实测口径 + pip-audit 结果；剩余差距行改指本报告 GAP-1；文档入口区补充 20260821 报告与归档清单链接。
2. `AGENTS.md:311-317`：头部"仍存技术债"注记修订——原注记 4 项中 3 项已闭环（Mamba fused 主路径、LongSequenceHandler 收敛、`_infer_activation_depth` 动态探测），`mypy: ignore-errors` 在 `src/` 内 0 命中（实为 `pyproject.toml` 两个模块级错误码豁免）；测试基线更新为 1957+107。
3. `docs/E2E训练与推理现状分析_2026-08-08.md`、`docs/r01_r03_systematic_repair_report_20260808.md`、`docs/E2E训练与推理代码修复报告_2026-08-08_v2.md`：3 份短页"当前权威分析"指向由 `project_analysis_20260816.md` 刷新为本报告。

---

## 2. 任务 1：未实现功能项识别与记录

### 2.1 识别方法

1. 以正式需求规范为对照基准：`.planning/REQUIREMENTS.md`（v2.1 24 条 + v2.2 10 条全部 `[x]` Complete，`grep -c "\[x\]"` = 24、`"\[ \]"` = 0）；`.planning/ROADMAP.md` Progress 表 18 个 Phase 全部 Complete（v1.0×6、v2.0×5、v2.1×3、v2.2×3，最后完成 2026-08-08）。
2. 排除已裁剪范围：2026-07-05 需求裁剪决定（实时质谱流 V2-02、自定义 PTM 数据库 V2-03、GUI、API key 扩展）不列为未实现（`AGENTS.md:297-299`；`tests/unit/test_roadmap.py` DeprecationWarning 守卫佐证）。
3. API 端点全量枚举比对：`grep "@router"` 于 `src/api/routes/*.py` + `src/api/monitoring.py`，共 **15 个端点**全部有实现与响应模型：

| 端点 | 位置 |
|---|---|
| `POST /predict`、`POST /predict/batch`、`POST /batch_predict`、`POST /predict/variant` | `src/api/routes/predictions.py` |
| `POST /initialize` | `src/api/routes/initialize.py` |
| `GET /health`、`GET /live`、`GET /ready`、`GET /model/info`、`GET /model_info` | `src/api/routes/model_info.py:21/59/81/164/165` |
| `POST /cross-scale/initialize`、`POST /cross-scale/predict`、`POST /cross-scale/batch_predict` | `src/api/routes/cross_scale.py` |
| `GET /health/detailed`、`GET /metrics` | `src/api/monitoring.py:23/129` |

4. 数据清单状态核对：`data/manifests/datasets.yaml` 中 `status: implemented` 与 `status: controlled` 分布逐项 grep（controlled 共 4 项：ptmatlas、proteometools_prospect、replogle、scgenescope）。
5. 未实现标记全库扫描：`TODO|FIXME|XXX` 在 `src/`+`scripts/`+`tests/` 中计数 **0**；`raise NotImplementedError` 仅 `src/models/encoders.py:19`（抽象基类，属设计）。

### 2.2 未实现项清单

**结论：不存在"已明确定义但完全未实现"的代码功能项或缺失 API 接口。** 剩余 2 项非代码功能缺口（与 0820 报告判定一致，本轮重新实证）：

| 编号 | 功能/接口名称 | 需求出处 | 优先级 | 影响范围 | 本轮证据 |
|---|---|---|---|---|---|
| GAP-1 | 真实数据科学验收流程（固定快照/指标阈值/验收报告，F-04） | `docs/CURRENT_STATUS.md` 剩余差距行；README 真实资产验收门禁说明 | **高** | **核心**（科学有效性验证） | `tests/real_assets/test_real_cptac_validation.py:43` `pytestmark = skipif(...)`，16 处 `PTM2CELLNET_RUN_REAL_ASSET_TESTS` 门禁；仓库无固定阈值/快照/验收报告文件。注意：`outputs/real_assets/` 存在 3 份 2026-07-06 的 ESM-3/DAVF 通过记录（`outcome: pass`），但 CPTAC 科学验收产物仍缺失，且现有记录已 >30 天不能反映当前状态 |
| GAP-2 | replogle / scgenescope 受控数据导入 | REQUIREMENTS DATA-01（受控资产显式登记即满足契约） | 中 | 边缘（数据侧） | `data/manifests/datasets.yaml` 二者 `status: controlled`；`scripts/` 无对应导入器；DATA-01 契约明确允许 controlled 显式登记，不构成工程违约 |

### 2.3 未闭环业务流程（GAP-1 主流程）

```mermaid
flowchart LR
    A[原始数据] --> B[manifest 登记]
    B --> C[导入器 bioplex/regnetwork/string/kinase/norman/scperturb]
    B -.replogle/scgenescope.-> X1[loader:null 受控仅登记 ♯GAP-2]
    C --> D[canonical 边表/表达 → 跨尺度 NPZ]
    D --> E[train_cross_scale.py]
    E --> F[/cross-scale/* API 已闭环 a63e4dd]
    F -.-> G[✗✗ 真实科学验收：门禁内 skip 无阈值/快照/验收报告 ♯GAP-1 ✗✗]
```

标准模型的离线训练→推理→API 链路与跨尺度链路均已闭环；唯"真实数据上的科学有效性验收"一步从未在启用门禁的状态下执行并沉淀报告。

### 2.4 已核销的"曾称未实现"项（防回归清单）

以下项在更早报告中曾被登记为未实现/待办，本轮逐一核销（关键证据）：

| 项 | 核销证据 |
|---|---|
| 跨尺度 API 三端点 | `src/api/routes/cross_scale.py` 三处 `@router.post`；路由接线 `src/api/routes/__init__.py:20` |
| TD-H02 embedding 预计算器 | `scripts/prepare_cross_scale_embeddings.py` + `tests/unit/scripts/test_prepare_cross_scale_embeddings.py` |
| 信号图原始格式导入 | `scripts/import_ppi_graphs.py`（Bioplex/RegNetwork/STRING） |
| scperturb h5ad 解析 | `scripts/import_scperturb.py`；manifest `status: implemented` |
| GSE90546 解析 | `scripts/import_norman_adamson.py` `parse_gse90546` + CLI `--parse-gse90546` |
| Mamba fused CUDA 路径 | `src/models/mamba_encoder.py:169` `_ssm_step_fused` 分支；验收锁定 `3da79c8` |
| 跨尺度 API 文档 | `docs/guides/deployment.md:141-157` |

---

## 3. 任务 2：未完全实现功能梳理

### 3.1 模块规模与测试分布（客观底数，本轮实测）

`src/` 各子包规模（行数）与对应单元测试用例数（`pytest --collect-only` 实测）：

| 子包 | 文件数 | 行数 | 直接测试用例数（unit 子目录） | 相关根级测试 |
|---|---|---|---|---|
| models | 39 | 14,182 | 48 | test_models*.py、test_mamba_encoder.py 等 |
| data | 29 | 9,091 | 67 | test_data*.py 系列 |
| training | 14 | 4,603 | 234 | test_trainers/callbacks/self_supervised 等 |
| api | 13 | 4,095 | 0（根级） | API 相关根级测试 **192 例** |
| integration | 11 | 3,305 | 136 | — |
| evaluation | 5 | 2,199 | 92 | — |
| analysis | 6 | 1,947 | 61 | — |
| utils | 10 | 1,984 | 0（根级） | test_utils/io/safe_io/config 等 |
| baselines | 2 | 644 | 5 | — |
| inference | 2 | 231 | 3 | — |

包间耦合（AST 解析相对导入实测）：`models` 出边 52（最高，主要指向 pretrained_encoders/utils/encoders/model_utils）、`api` 出边 39（utils 15）、`data` 出边 37（utils 15）、`training` 出边 22；`utils` 为最常被依赖的基础层（35 文件跨包引用），层次结构总体健康（无反向依赖环证据）。

### 3.2 模块完成度总表

完成度为综合文件规模、契约覆盖、测试覆盖与文档证据的估计值（±3%）：

| 模块 | 完成度 | 判定依据摘要 | 缺失组件/依赖 |
|---|---|---|---|
| `src/data/` | 97.5% | DataLoader mixin 组装、DataPreprocessor 三契约方法、FeatureExtractor、PTMDataset、DAVFSiteAugmenter、homology_splitter、data_manifest/data_contract | 蓝图类名 `PTMDataModule` 不存在（被 4 个具体 DataModule 取代，命名偏差） |
| `src/models/` | 95.0% | 全编码器（CNN/LSTM/GRU/Transformer/Mamba）、PTMEmbedding/PTMAttention、PTM2CellNetBase/PTM2CellNet/Large、DAVF 全链路、ensemble、variant_effect、signaling_network、geneformer | Mamba 无 CUDA 时串行回退；ESM-2/ProtT5/scVI 可选依赖 |
| `src/training/` | 95.5% | Trainer、Focal/MultiTask 损失、configure_optimizer、ModelCheckpoint/EarlyStopping、artifacts.py、self_supervised | `self_supervised.py` 单测覆盖 31.62%（`docs/TEST_COVERAGE.md:57`）——测试缺口而非实现缺口 |
| `src/evaluation/` | 98.0% | 分类/回归/排序指标、独立 `evaluate`/`cross_validate`、explainers、visualization | 无 |
| `src/utils/` | 98.0% | SafeUnpickler + safe_torch_load 分层防御、setup_logger、Config、helpers、checkpoint_utils | 无 |
| `src/api/` | 96.0% | 工厂 + 限速/Prometheus/请求体限制/CORS/API-key 中间件、15 端点全实现、192 例专项测试 | 无 |
| `src/analysis/` + `src/integration/` | 92.5% | variant_workflow/parser/gene_mapper/pathway_integration + pdc_client；GenKI 全套；ptm_virtual_perturbation + contracts | KEGG/Reactome 完整集成为需求显式 out-of-scope（REQUIREMENTS 豁免）；GenKI 离线阵列后端读 mock 文件 |
| `scripts/` | 92.0% | 41+ CLI 齐全（含 8 月新增 5 个导入脚本）；126 例专项测试 | 上轮报告的 pmads_ridge TODO 已消失（本轮 grep 计数 0） |
| `src/inference/` | 95.0% | cross_scale_predictor + 契约测试 | 规模小（231 行），覆盖 3 例偏薄 |
| `src/baselines/` | 90.0% | PMADS Ridge 基线 + CLI + 5 例测试 | 功能单一，符合 BASE-01/02 最小契约 |

**整体加权完成度约 95.4%**（按行数加权）。代码卫生实测：`src/`+`scripts/`+`tests/` TODO/FIXME/XXX 计数 **0**（较 0820 报告的"1 处 TODO"进一步改善）；零 `raise NotImplementedError`（唯一一处为抽象基类设计）。

### 3.3 与需求规范的差异明细

| 需求/规范原文 | 实际实现 | 差异性质 |
|---|---|---|
| AGENTS 蓝图 `class PTMDataModule`（§数据模块接口契约） | `PTMLightningDataModule`、`PTMPlainDataModule` 等 4 个具体 DataModule | 类名契约不符，功能全覆盖，属命名偏差 |
| CUDA Mamba 应可用（环境齐备时） | 无 CUDA/未装 mamba_ssm 时走串行实现（`mamba_encoder.py:169` 条件分支） | 性能缺陷（非正确性）；主路径 fused/parallel 已实现并有验收测试 |
| SignalingNetworkMapper KEGG/Reactome 集成 | 失败时降级为内置通路集 | 被 REQUIREMENTS 与 CONF-03 显式豁免（sanctioned 降级） |
| "禁止隐式随机 fallback"（AGENTS 工程规范） | `geneformer_embedding.py` 随机嵌入 fallback（可被 strict_assets 关闭）+ cross_scale 响应 `fallback_flags` 显式上报 | 已显式化、风险受控；非严格模式仍可能产出随机嵌入预测 |
| Preprocessor/FeatureExtractor 契约签名 | `clean_sequences` 增加 `sequence_col` 参数、`split_dataset` 签名扩展 | 向前兼容但非严格一致 |

### 3.4 三分类清单（含判断标准）

**一、部分实现但可用**——判断标准：核心路径可运行且有替代路径，缺口有明确文档登记。
- YAML 数据清单：required 数据集中 replogle/scgenescope 2 项 controlled（其余 implemented）。
- 真实资产验收未启用（GAP-1）："全绿"≠科学验收；`outputs/real_assets/` 仅有 3 份 2026-07-06 的 ESM-3/DAVF 记录，CPTAC 验收缺失。

**二、实现但有缺陷**——判断标准：代码可运行且测试通过，但存在已登记的性能/依赖/测试缺口。
- Mamba 串行回退（无 CUDA 时性能损失，属设计而非错误）。
- `pip check` 可选依赖冲突（scgpt 0.2.4 vs scvi-tools 1.4.3 并存，F-10）。
- `self_supervised.py` 单测覆盖 31.62% 偏低（回归风险）。
- mypy 双版本口径差（TD-N-06，见 §4.3）。
- 锁定依赖含 64 个已知漏洞条目（TD-N-08，本轮新增）。

**三、实现但不符合规范**——判断标准：行为语义等价但类名/签名/存放位置与契约文字不一致。
- `PTMDataModule` 蓝图类名未提供（§3.3）。
- 根 `signaling_network.py` 为 re-export shim（CODE-03 已豁免）。

---

## 4. 任务 3：技术债识别、分类与解决策略

> 已先读 0820 报告及 `archive/20260816`、`archive/20260820` 各 MANIFEST 建立"已提及"基线，仅报告增量与状态变化。

### 4.1 分级标准

| 级别 | 标准 |
|---|---|
| 严重 | 影响正确性或安全性，直接引发错误数据/错误行为，或阻塞主要功能 |
| 高 | 显著损害可维护性、安全性或合规性，未来大概率引发事故 |
| 中 | 明显增加维护成本，但当前不产生错误 |
| 低 | 风格/整洁层面问题 |

### 4.2 存量技术债状态复核（对照 0820 报告 7 项）

| 编号 | 上轮判定 | 本轮实测 | 状态 |
|---|---|---|---|
| TD-N-01 legacy_loaders 死代码 | 中 | `src/data/loaders/legacy_loaders.py` 仍存在，全仓零调用（唯一引用为 `roadmap.py:10` docstring） | **维持（中）** |
| TD-N-02 lazy_import 零测试 | 中 | `tests/unit/test_lazy_import.py` 仍不存在 | **维持（中）** |
| TD-N-03 deployment 文档缺 /predict/variant | 中 | `grep -c variant docs/guides/deployment.md` = **0** | **维持（中）** |
| TD-N-04 Pooled 变体重复 | 低 | `encoders.py:186/223/254` 三个 Pooled 类仍同构 | **维持（低）** |
| TD-N-05 iterrows 14 处 | 低 | 实测仍 14 处，分布 9 文件（explainers 4、multitask_dataset/pathway_integration 各 2），均非训练热路径 | **维持（低）** |
| TD-N-06 mypy 类型债 | 中（25 errors 口径） | 双版本实测：mypy 1.20 → 25E/12F；mypy 2.1 → 5E/2F。**范围修正**：以 1.20 口径为清偿超集 | **修正（中）** |
| TD-N-07 根目录垃圾文件 | 低 | `=1.0.0`/`=3.0.0` 内容确认为 pip 误重定向日志、`0.7692`/`EOF`/`=0.8.0`/`=2.5.0` 为空文件；`x.pt` 为 14KB 测试 checkpoint、zip 为数据压缩包、omp-session html 为会话产物——均不入库（gitignore 白名单外） | **维持（低），归属已查明** |

另核实 AGENTS.md 头部注记 4 项存量债：3 项闭环 + 1 项描述失实（本轮已直接修订该注记，衍生文档债就地清偿）。

### 4.3 新增技术债清单

| 编号 | 类别（SonarQube 参照） | 严重度 | 问题 | 证据 |
|---|---|---|---|---|
| TD-N-08 | Vulnerable dependency（Security hotspot） | **高** | `requirements-lock.txt` 含 **64 个已知漏洞条目 / 12 包**：pillow 12.1.1（26 条 PYSEC，修复版 12.3.0）、aiohttp 3.13.5（14 条）、sqlparse 0.5.5（4 条，修复 0.6.0）、urllib3 2.6.3（3 条，修复 2.7.0）、transformers 4.57.6（3 条，修复 5.x）、setuptools 80.10.2（2 条，修复 83.0.0）、idna/msgpack/datasets/lightning/mamba-ssm/uv/pytorch-lightning 各 1–3 条；其中 **pytorch-lightning 1.9.5 有 3 条漏洞且与 lightning 2.6.5 双包共存**（PYSEC-2026-1857 无修复版本）；torch 系 6 个 +cu118 包因 PyPI 无对应版本无法审计 | `pip-audit -r requirements-lock.txt --disable-pip --no-deps` 全量输出（2026-08-21 实测）；`requirements-lock.txt:195` `pytorch-lightning==1.9.5` 与 `:96` `lightning==2.6.5` 并存；`pip show` 证实二者均安装（后者 Required-by scvi-tools） |
| TD-N-09 | Test infrastructure | 低 | `pytest --cov` 与 torch 2.4.1 在收集期冲突：带 `--cov` 运行任何导入 torch 的测试即报 `RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace _inductor_test`（不带 cov 全绿）；覆盖率门禁只能靠 CI 独立环境或历史基线（75.27%，2026-08-09） | 本轮 4 次复现（test_aa_constants / test_self_supervised × --cov 组合），清 inductor 缓存与 `TORCHDYNAMO_DISABLE=1` 均无效；无 cov 时同批测试 26 passed |

**未发现新的"严重"级技术债**。TD-N-08 升为当前最高优先技术债（安全暴露面随部署面扩大）。

### 4.4 高/中级技术债解决策略

#### TD-N-08 锁定依赖漏洞清偿（高）

- **影响范围**：所有经 `requirements-lock.txt` 部署的环境。pillow/aiohttp/urllib3 属网络/图像攻击面（API 服务直接暴露 aiohttp+urllib3）；transformers 漏洞涉及模型加载路径；pytorch-lightning 1.9.5 的 3 条漏洞中 1 条无上游修复版本，且该包仅为 lightning 2.6.5 的传递依赖（源码自身仅做字符串检测，`src/api/routes/initialize.py:162`）。
- **方案 A（推荐，分两批）**：
  - 第一批（0.5 天）：升级有干净修复版的 7 包——pillow→12.3.0、sqlparse→0.6.0、urllib3→2.7.0、idna→3.15、msgpack→1.2.1、datasets→5.0.1、setuptools→83.0.0、uv→0.11.15；跑全量单测回归。
  - 第二批（1.0 天）：处理 transformers 5.x 大版本跳跃（API breaking 风险高，需单独回归 PLM 资产测试）与 pytorch-lightning 双包问题（评估 lightning 2.6.5 是否已不拉入 1.9.5，或 pin 至无漏洞传递路径）。
- **方案 B**：仅升级 pillow/aiohttp/urllib3 三个网络攻击面包（0.5 天），其余挂 `--ignore-vuln` 白名单并登记复审日期。优点：改动面最小；缺点：transformers 攻击面保留。
- **方案 C**：引入 `pip-audit` 进 CI nightly（0.5 天）+ 按月 ratchet，先建机制后清偿。可与 A/B 叠加。
- **实施资源**：熟悉 Python 依赖生态的工程师 1 名；无需额外工具（pip-audit 已验证可用）。**风险**：transformers 5.x 可能破坏 pretrained_encoders 契约测试（已有 85998f7 契约套件兜底）；pillow 大版本内升级风险低。

#### TD-N-06 mypy 类型债清偿（中，范围修正）

- **影响**：`Tensor | Module` 联合误标注掩盖真实类型约束；双版本口径差使"0 errors"类自述不可信。
- **方案 A（推荐，分两批）**：第一批 0.5 天修两版本共同报错的 5 处（`predictions.py:666/770`、`cross_scale.py:466/706/714`——`str | None` 收窄加显式分支或断言、`warnings_list` 加注解）；第二批 1.0 天修 mypy 1.20 特有的 20 处（`peft_config.py` 5、`long_sequence.py` 4、`ensemble.py`/`pmads_ridge.py`/`cross_scale_trainer.py` 各 2 等，根因为 forward 引用注解不精确）。修后将 `mypy==<pinned>` 写入 dev 依赖固定口径。
- **方案 B**：升级 CI 到 mypy 2.1 口径（5 errors），旧版差异记录存档。优点：工作量降 80%；缺点：本地多环境口径仍分裂。
- **方案 C**：追加 `disable_error_code` 豁免。缺点：掩盖真实约束，不推荐。
- **实施**：A 合计 1.5 工作日；B 合计 0.5 工作日。资源：熟悉 PyTorch 类型生态工程师。风险：`cast` 滥用引入假阴性，需配合单测抽查。

#### TD-N-01 legacy_loaders 死代码删除（中）

- **影响**：为已取消 roadmap 需求保留约 250 行僵尸实现，徒增维护面。
- **方案 A（推荐）**：删除模块 + `roadmap.py` docstring 改"已移除"。全仓 grep 已确认零代码引用。0.5 天，风险极低。
- **方案 B**：迁入 docs/archive 作历史样本。可追溯但无人维护。
- **方案 C**：`@deprecated` 延迟删除。无硬 deadline，不推荐。

#### TD-N-02 lazy_import 补测（中）

- **影响**：evaluation/analysis 两包可选依赖垫片的失败语义若退化将静默变化。
- **方案 A（推荐）**：新增 `tests/unit/test_lazy_import.py`——成功加载、ImportError 重抛、attr 缺失抛 AttributeError、缓存命中不二次 import。0.5 天，无风险。
- **方案 B**：改 `find_spec` 前置探测。改动大收益低，不推荐。

#### TD-N-03 deployment 文档补录 /predict/variant（中）

- **影响**：端点参数与 400/503/504 错误语义无公开文档。
- **方案 A（推荐）**：deployment.md 增加"变体预测端点"小节（请求/响应/错误码 + 与 /predict 差异），与 `a0cfb61` 拆分后行为核对。0.5 天。
- **方案 B**：仅依赖 OpenAPI /docs 自动生成。离线不可见，与现有文档风格不一致。

**低级债处理建议**：TD-N-04 用 `create_pooling_layer` 工厂收敛三个 Pooled 变体（1 天，并入下次 encoder 改动）；TD-N-05 保持观察（非热路径）；TD-N-07 清理根目录垃圾文件（0.5 天，`x.pt`/zip 先确认无归属再删）；TD-N-09 在 CI 固定 coverage 环境（Ubuntu + pinned torch）规避本地冲突，或升级 torch 后复测（0.5 天观察项）。

---

## 5. 结论与建议

1. **需求符合度**：正式需求规范定义的代码功能全部实现（24/24 需求条目、18/18 Phase、15/15 API 端点），工程完成度整体约 95.4%。最高优先事项仍是 **GAP-1 真实数据科学验收（F-04）**——建议定义固定数据快照、指标阈值与验收报告模板并接入 `tests/real_assets/` 门禁（预计 2–3 工作日，需真实数据授权）；GAP-2 维持 controlled 登记即可。
2. **本轮最重要的两个新发现**：
   - **mypy 双版本口径差**（25E vs 5E）：0820 报告数字可复现但未注明版本，导致跨会话不可比。建议立即 pin mypy 版本并按 §4.4 方案 A 清偿。
   - **TD-N-08 依赖漏洞（高）**：64 个漏洞条目首次浮出水面，其中网络攻击面三件套（pillow/aiohttp/urllib3）均有干净修复版，第一批 0.5 天即可清掉大半；建议同步把 pip-audit 纳入 nightly CI。
3. **质量门禁实况**：compileall / 单测 1957+6 / 集成 107+1 / ruff 全绿；`pip check` 4 条已知可选冲突维持登记。coverage 门禁受 TD-N-09 影响，本地无法直接复测 branch coverage，下一轮应在 CI 或独立环境刷新 75.27% 基线。
4. **文档治理**：本轮归档 1 份（0820 报告）、修订 AGENTS.md 过时注记、刷新 CURRENT_STATUS 与 3 份短页指向。文档链路（CURRENT_STATUS → 最新报告 → archive MANIFEST）现已一致。
5. **技术债总量**：活跃债 **0 严重 / 1 高 / 4 中 / 4 低**。高+中级合计约 4.0 工作日可全部清偿（TD-N-08 1.5 + TD-N-06 1.5 + TD-N-01/02/03 各 0.5，取方案 A 口径）。
6. **版本控制**：本地领先远程 26 个提交，建议择机 `git push origin main` 发布（本次任务范围限定本地操作，未推送）。

---

## 6. 任务执行统计

### 6.1 子智能体调用统计

| 智能体名称 | 类型 | 调用次数 | 主要执行任务 | 平均执行时长 |
|---|---|---|---|---|
| （无） | — | **0 次** | 本轮全部工作由主代理直接执行：版本控制五项验证、pip-audit 首审、mypy 双版本根因定位、归档落地、文档刷新、三项系统性复核与报告撰写 | — |

**简要分析**：与 0820 轮（3 个并行子代理、墙钟 9.3 分钟）不同，本轮采用主代理直执模式。原因：(1) 上一轮报告提供了完整的模块映射与技术债基线，增量复核的检索面大幅缩小；(2) 本轮的关键结论（mypy 版本差异、pip-audit 结果）需要精确的跨环境对照实验，拆给子代理反而丢失上下文；(3) 归档与文档修订是强顺序依赖操作，无可并行切片。代价是主上下文消耗更高，但换来了每个结论的直接实测证据链（所有数字均可由附录 A 命令复现）。

### 6.2 工作量分布

| 阶段 | 主要动作 | 关键产出 |
|---|---|---|
| 前置操作 | git fetch/status/log 核对、compileall、单测、集成测试、ruff、mypy×2 环境、pip check、pip-audit | 验证矩阵（§1.2）、mypy 根因定位 |
| 归档 | git mv 归档 0820 报告、MANIFEST/README/重定向短页、AGENTS/CURRENT_STATUS/3 短页刷新 | `archive/20260821/` 全套 |
| 任务 1 | REQUIREMENTS/ROADMAP 逐条核对、15 端点枚举、TODO/NotImplemented 全库扫描、GAP 复核 | §2 清单 |
| 任务 2 | 10 模块规模/测试分布实测、包耦合 AST 分析、完成度评估 | §3 总表 |
| 任务 3 | 存量 7 项逐项复核、pip-audit 增量识别、解决策略制定 | §4 清单 |

---

## 附录 A：验证命令与结果汇总

```text
$ git fetch origin && git rev-list --left-right --count main...origin/main
26 0

$ python -m compileall -q src scripts; echo $?
0                                    # COMPILEALL_OK

$ python -m pytest tests/unit -q
1957 passed, 6 skipped in 115.28s    # 与 0820 基线一致

$ python -m pytest tests/integration -q
107 passed, 1 skipped in 740.19s     # 与 0820 基线一致

$ ruff check src scripts tests
All checks passed!

$ python -m mypy src                 # SSH_unit env, mypy 2.1.0
Found 5 errors in 2 files (checked 134 source files)

$ /home/scu/anaconda3/bin/python -m mypy src   # base env, mypy 1.20.0
Found 25 errors in 12 files (checked 134 source files)

$ pip-audit -r requirements-lock.txt --disable-pip --no-deps
64 vulnerability entries across 12 packages（pillow 26 / aiohttp 14 / sqlparse 4 /
urllib3 3 / transformers 3 / setuptools 2 / idna 2 / pytorch-lightning 3 /
msgpack 1 / datasets 1 / lightning 1 / mamba-ssm 1 / uv 1）
（torch 系 6 个 +cu118 包无法审计：PyPI 无对应版本）

$ grep -rn "TODO\|FIXME\|XXX" src/ scripts/ tests/ --include="*.py" | wc -l
0

$ grep -rn "raise NotImplementedError" src/ --include="*.py"
src/models/encoders.py:19            # 抽象基类，设计使然
```

---

## 附录 B：报告维护惯例（供下一轮接替者）

- 本报告发布后即为当前权威分析；下一份权威报告产生时，请将本文件 `git mv` 至 `archive/YYYYMMDD/reports/` 并在原路径留重定向短页。
- mypy 数字必须注明版本号（本轮教训：0820 报告未注明导致口径混乱）。
- pip-audit 结果随 lock 文件变化快速腐化，引用时注明审计日期与 lock 版本。
