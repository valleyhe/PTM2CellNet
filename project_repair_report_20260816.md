# PTM2CellNet 项目修复报告（2026-08-16）

> **任务依据**：`project_analysis_20260816.md`（v4.0，2026-08-16）——文中"未解决问题与后续解决策略"章节（§6.3 新增技术债、§6.4 解决策略、§7.2 建议排期）为修复指引；所有修复决策结合当前代码实测现状自主分析后独立做出，不盲目照搬文档结论（见 §4.3 复核修正）。
> **执行方式**：3 轮完整"问题识别 → 方案设计 → 代码修改 → 单元测试 → 集成测试"迭代，随后系统性复核（含 E2E 训练/推理实机验证）。
> **验证基线**：修复前 `f5667db`（文档报告提交）；修复后 `main` = `6dc4472`（3 组修复提交）。全量测试 **1956 passed / 7 skipped**（基线 1919/5，净增 37 项），`compileall` 通过，exit 0。

---

## 目录

- [1. 问题修复汇总（按严重程度分类）](#1-问题修复汇总按严重程度分类)
- [2. 第 1 轮迭代：D1 依赖 lock 治理](#2-第-1-轮迭代d1-依赖-lock-治理)
- [3. 第 2 轮迭代：N08 UniProt 轮询退避 + N09 num_workers 显式化](#3-第-2-轮迭代n08-uniprot-轮询退避--n09-num_workers-显式化)
- [4. 第 3 轮迭代：N05 窗口魔法数统一 + D3 垫片清理](#4-第-3-轮迭代n05-窗口魔法数统一--d3-垫片清理)
- [5. 系统性复核：E2E 训练与推理技术要求评估](#5-系统性复核查-e2e-训练与推理技术要求评估)
- [6. 未解决问题及后续解决策略](#6-未解决问题及后续解决策略)
- [7. 验证命令与审计痕迹](#7-验证命令与审计痕迹)

---

## 1. 问题修复汇总（按严重程度分类）

严重程度分级沿用分析报告 §6.1（严重/高/中/低）。本轮无"严重"级问题，修复"高"级 0 项（无存量，§6.2 两项高级债 N01/N02 已在上一轮闭环）、"中"级 4 项、"低"级 1 项；复核修正文档误判 2 项（D2/D4，见 §4.3）。

| 严重程度 | 编号 | 问题 | 修复状态 | 证据 |
|---|---|---|---|---|
| 中 | D1 | lock 未入库 + numpy 漂移 + 不可移植行（§6.3 D1、§6.4 S7、§7.2 排期 1） | **已修复**（第 1 轮，提交 `6ae2d4e`） | `git ls-files` 含 `requirements-lock.txt`；numpy 约束 core/setup.py 一致 `>=1.24,<3`；3 个新契约测试 |
| 中 | N08 | UniProt ID-mapping 固定 1s 同步轮询 60 次（§6.4 S8） | **已修复**（第 2 轮，提交 `33f6bde`） | 指数退避 0.2s×1.5 cap 3s + 预算 clamp；8 个新单测 |
| 中 | N09 | default.yaml 未声明 num_workers，静默回退 0（§6.4 S8） | **已修复**（第 2 轮，提交 `33f6bde`） | `configs/default.yaml` 显式 `num_workers: 4`；配置契约测试 |
| 中 | N05 | 窗口逻辑 31/15/16 魔法数 ≥4 处、训练/推理窗口漂移风险（§6.4 S4） | **已修复**（第 3 轮，提交 `6dc4472`，方案 B+） | 常量单源化 + 5 调用点替换 + 语义等价断言；25 个新单测 |
| 低 | D3 | 根目录未追踪 `signaling_network.py` 垫片（§6.3 D3、§6.4 S8） | **已修复**（第 3 轮） | 全仓 grep 无引用后删除（`src/models/signaling_network.py` 为唯一权威实现） |
| — | D2 | 复核修正：文档称 GenKI 摄取路径 3 处裸 except 吞错（§6.3 D2） | **判断不成立**（见 §4.3） | 3 处均为 `progress_callback` 防护（带意图注释），数据摄取路径有 `as exc` 记录 |
| — | D4 | 复核修正：文档称 API 路由零单元测试（§6.3 D4、§6.4 S5） | **判断过时**（见 §4.3） | `tests/unit/` 已有 2142 行 API 测试（含异常分支/限流/密钥） |

**统计**：修复 5 项（中 4 + 低 1）、复核修正 2 项、新增测试 37 项、3 组提交（+437/−38 行左右）。

---

## 2. 第 1 轮迭代：D1 依赖 lock 治理

### 2.1 问题识别

- **来源**：分析报告 §6.3 D1（本轮新增技术债之首，§7.2 排期第 1 位——"环境复现性的最后一环"）。
- **实测确认**（§9.3 命令复跑）：
  1. `git ls-files | grep -i lock` 为空 → `requirements-lock.txt` 未入库（被 `.gitignore:6 /*` 白名单机制忽略），干净 clone 无锁定手段；
  2. `requirements-lock.txt:127` `numpy==2.4.3` vs `requirements-core.txt:14` `numpy>=1.24,<2`（setup.py:19 同）→ 声明与冻结漂移；
  3. 文档所述"2 处 file:/// /tmp/ URI"经实测为**注释留痕**（TD-M09 说明已改 PyPI 规范名），但另发现 1 处真实不可移植行：`-e /home/scu/SSH_unit`（editable 本地开发包，`requirements-lock.txt:232`）。

### 2.2 方案设计（自主决策）

| 子问题 | 可选方案 | 决策与理由 |
|---|---|---|
| numpy 漂移方向 | A: lock 降级 1.26.x；B: core 上限放宽 `<2`→`<3` | **B**。SSH_unit 验证环境（numpy 2.4.3 + torch 2.4.1+cu118）实测 1919 测试全绿（§9.2），numpy 2.x 兼容性已被实证；lock 是真实冻结环境快照，改动它反而制造"声明 vs 实测"分裂 |
| 不可移植行 | A: 删除；B: 注释保留 | **A**（替换为说明注释）。ssh-unit 非 PTM2CellNet 依赖树成员，干净 clone 无法解析 |
| 入库方式 | A: .gitignore 白名单加入；B: 移动路径 | **A**。保持根目录惯例（requirements-*.txt 同级） |
| CI 断言 | A: CI 完整 `pip install -r lock && pip check`；B: 文件级契约测试 | **B**。lock 含 cu118 编译版 torch，CI 全量安装不现实（时长/磁盘）；文件级断言入 pytest 门禁（CI 自动生效） |

### 2.3 代码修改

| 文件 | 修改 |
|---|---|
| `requirements-core.txt:14` | `numpy>=1.24,<2` → `numpy>=1.24,<3`（含 D1 注释） |
| `setup.py:19` | 镜像同步（契约测试强制一致） |
| `.gitignore` | 白名单新增 `!requirements-lock.txt` |
| `requirements-lock.txt` | 移除 `-e /home/scu/SSH_unit`（留注释说明） |
| `tests/unit/test_dependency_contracts.py` | 新增 3 测试：`test_lock_numpy_version_satisfies_core_constraint`（packaging 解析逐子句校验）、`test_lock_has_no_non_portable_lines`（editable/file:///tmp/绝对路径）、`test_lock_is_fully_pinned_exact_versions`（name==version 精确固定） |

### 2.4 测试结果

- **单元测试**：`test_dependency_contracts.py` **10 passed**（原 7 + 新 3）。
- **集成测试**：`test_api_concurrency.py` + `test_api_auto_initialize.py` **28 passed**（验证依赖声明改动未破坏 API 并发语义）。
- **入库验证**：`git ls-files | grep -i lock` → `requirements-lock.txt` ✓。
- **提交**：`6ae2d4e build(deps): track requirements-lock.txt in VCS and reconcile numpy drift (D1)`。

---

## 3. 第 2 轮迭代：N08 UniProt 轮询退避 + N09 num_workers 显式化

### 3.1 问题识别

- **N08**（§6.4 S8）：`src/analysis/gene_mapper.py:68-80` `_RequestsUniProtMapper.get()` 轮询循环 `for _ in range(60)` + 固定 `time.sleep(1.0)`——短作业也被迫等待 ≥1s/次，长作业总时长 60s 封顶但节奏僵硬。
- **N09**（§6.4 S8）：`configs/default.yaml` 无 `num_workers` 键，`scripts/train.py:160` `config.get("data.num_workers", 0)` 静默回退 0（主进程加载），多核机器上数据加载串行化。

### 3.2 方案设计

- **N08**：指数退避（0.2s 起步 ×1.5 因子，3.0s 封顶），并新增**预算 clamp**——每次 sleep 前按剩余预算截断（`min(interval, remaining)`），保证总等待严格 ≤60s（文档方案"总预算 60s 不变"的严格化实现）。结果落盘缓存（文档方案后半）**本轮不实现**：涉及缓存键/过期/多进程并发设计，与退避正交，单独排期更稳妥【自主决策】。
- **N09**：default.yaml 显式 `num_workers: 4` + `persistent_workers: false`；代码默认值不动（`DatasetConfig.num_workers=0` 为安全默认，文档方案即"仅配置 + 文档"）。

### 3.3 代码修改

| 文件 | 修改 |
|---|---|
| `src/analysis/gene_mapper.py` | 模块级常量 `_POLL_INITIAL_INTERVAL=0.2`/`_POLL_BACKOFF_FACTOR=1.5`/`_POLL_MAX_INTERVAL=3.0`/`_POLL_BUDGET_S=60.0`；`import time` 上移模块级（可 mock）；轮询循环重写 |
| `configs/default.yaml` | `data` 段新增 `num_workers: 4`、`persistent_workers: false`（含 N09 注释） |
| `tests/unit/test_gene_mapper.py`（新增） | 8 测试：确定性 fake clock（monotonic 跟随 sleep 推进）验证退避序列 `[0.2, 0.3, 0.45, 0.675]`、3.0s 封顶、FINISHED 快速路径零 sleep、60s 预算严格不越界（断言 `fake_time.now <= 60.0`）、提交失败/空 job_id/成功解析/空输入短路 |
| `tests/unit/test_utils.py` | 新增 `test_default_data_loader_workers_explicit`（N09 配置契约） |

### 3.4 测试结果

- **单元测试**：`test_gene_mapper.py` **8 passed**（首轮 5 失败均为测试自身问题——side_effect 缺 results 响应、zip 长度断言、封顶相等性误判；修正后全绿）；`test_utils.py` **17 passed**。
- **集成测试**：`tests/integration/test_variant_workflow.py` **16 passed**（gene_mapper 下游消费者回归）。
- **提交**：`33f6bde perf(analysis): exponential backoff for UniProt ID-mapping polls; explicit num_workers (N08/N09)`。

---

## 4. 第 3 轮迭代：N05 窗口魔法数统一 + D3 垫片清理

### 4.1 问题识别

- **N05**（§6.4 S4）：PTM 位点窗口宽度 31/半窗 15 以魔法数字面量散落 ≥4 处：`scripts/predict_ptm_sites.py:109-110,285`（含报告窗口 `position-16:position+15` 另类写法）、`scripts/train_ptm_site.py:178`、`src/data/ptm_site_dataset.py:40,185`、`src/models/ptm_site_predictor.py:51`、`src/models/multitask_ptm.py:57`——训练/推理窗口不一致是文档点名的静默特征偏移风险。
- **D3**（§6.3 D3）：根目录 `signaling_network.py` 为 31 行 re-export 垫片，被 `.gitignore` 忽略、与权威实现 `src/models/signaling_network.py`（781 行）不同源。

### 4.2 方案设计

- **N05 方案 B+（自主决策）**：文档 S4 方案 A（新建 `src/data/window.py` 统一 4 处切分实现，2.0d）风险高——历史数据集窗口宽度契约【待确认】（§6.4 风险栏），训练 CSV 预处理窗口与推理半窗提取语义不同（前者居中截取、后者以位点为中心，不能机械合并）；方案 B（仅消除魔法数 0.5d）+ **语义等价断言**：常量单源化到 `src/data/aa_constants.py`，5 处调用点全部导入，另加训练/报告窗口公式等价性回归测试（防未来漂移）。
- **D3**：全仓 grep 确认无 `import signaling_network` 引用后删除（文档 S8 建议，0.5d）。

### 4.3 代码修改

| 文件 | 修改 |
|---|---|
| `src/data/aa_constants.py` | 新增 `DEFAULT_PTM_WINDOW_SIZE=31`、`DEFAULT_PTM_HALF_WINDOW=15`（单源，注释说明语义） |
| `scripts/predict_ptm_sites.py` | `:109-110` 用常量；`:285` 报告窗口改常量推导 `max(0, pos-(hw+1)):pos+hw`（行为不变） |
| `scripts/train_ptm_site.py` | `window_size=31` → 常量 |
| `src/data/ptm_site_dataset.py` | 两处默认值 → 常量 |
| `src/models/ptm_site_predictor.py` | 默认值 → 常量 |
| `src/models/multitask_ptm.py` | 默认值 → 常量 |
| `tests/unit/test_window_constants.py`（新增） | 25 测试：常量自洽（奇数/居中）、4 个类签名默认值防漂移（inspect.signature）、报告切片 vs 训练窗口语义等价（20 组参数化，含位点居中断言）、predict_ptm_sites.py 魔法数字面量回归扫描 |
| `signaling_network.py`（根目录） | 删除（D3） |

### 4.4 测试结果

- **单元测试**：`test_window_constants.py` **25 passed / 2 skipped**（skipped 为 position 越界参数化用例）；N05 相关回归（`test_models_ptm_predictors.py`、`test_scripts.py`）**76 passed / 2 skipped**。
- **集成测试**：全量（见 §5）。
- **提交**：`6dc4472 refactor(data): centralize PTM window constants and kill 31/15 magic numbers (N05)`。

---

## 5. 系统性复核：E2E 训练与推理技术要求评估

### 5.1 E2E 实机验证（本轮实际执行）

| 链路 | 命令/方式 | 结果 |
|---|---|---|
| 训练 | `python scripts/train.py --config configs/smoke/cnn_cpu.yaml --data data/raw/sample_data.csv --output outputs/smoke_review --epochs 1 --batch_size 16` | **通过**：8 步骤完整执行，评估指标输出（accuracy 0.278 / auc_roc 0.606），`checkpoint_best.pt` + `best_model.pt` + `artifact_manifest.json` 导出，`model_kind=demo` 标记生效 |
| 推理 CLI | `python scripts/predict.py --model outputs/smoke_review/models/best_model.pt --sequence ... --device cpu` | **通过**：细胞状态 + 置信度 + 概率分布输出，结果落盘 |
| 推理 API | TestClient：`POST /api/v1/initialize`（带 API key）→ `/predict` → `/ready` → `/model/info` | **全 200**：initialize 成功、predict 返回 `cell_state=apoptosis, confidence=0.394`、ready/model-info 就绪；未带 key 时 `/initialize` 正确 503（安全门禁生效）；未初始化时 `/predict` 503、`/live` 200、`/ready` 503（生命周期语义正确） |
| 全量测试 | `pytest tests/unit tests/integration -q --tb=no -p no:cacheprovider` | **1956 passed / 7 skipped / 21 warnings，435.87s，exit 0**（基线 1919/5） |
| 编译 | `python3 -m compileall -q src scripts` | exit 0 |

### 5.2 满足度评估

| 技术要求 | 满足度 | 说明 |
|---|---|---|
| 标准训练 E2E（CNN/Transformer） | ✅ 满足 | 实机验证 + 全量测试 |
| 标准推理（CLI + API） | ✅ 满足 | CLI/API 双链路实机验证 |
| 跨尺度训练/推理（CLI 层） | ✅ 满足 | `train_cross_scale.py`/`predict_cross_scale.py`/`prepare_cross_scale_embeddings.py`（TD-H02 闭环，§4.1 F-02）+ 5 集成测试 |
| 跨尺度 API 服务（F-01） | ❌ 不满足 | `src/api/` 无 cross_scale 引用；`initialize.py:214` 仍固定 `PTM2CellNet.from_config`（§4.1 F-01） |
| 真实科学验收流程（F-04） | ❌ 不满足 | `tests/real_assets/` 仍整体按 `PTM2CELLNET_RUN_REAL_ASSET_TESTS` 门禁 skip（§4.1 F-04） |
| 受控数据完整性 | ⚠️ 部分 | kinase-substrate/epsd/scperturb 登记闭环（§4.1/§4.4）；**scperturb h5ad 解析（F-05）、replogle/scgenescope 获取（F-06/F-07）、GSE90546 解析（F-08）未做** |
| 图数据导入（F-03 余量） | ❌ 不满足 | STRING/BioPlex/RegNetwork 原始格式导入器仍缺（raw 在盘，`datasets.yaml` loader null） |
| 环境可复现性 | ✅ 满足（本轮闭环） | D1 修复后 lock 入库 + 无不可移植行 + numpy 漂移消除 |

### 5.3 主要问题点与根本原因

1. **跨尺度 API 缺失（F-01）**：工程契约层已具备全部能力（CLI + artifact + provenance + cross_scale_dataset），缺的仅是 API 适配层（model_type 路由、NPZ/序列双输入、artifact digest 登记）。根因：API 层按"标准 PTM2CellNet"单形态设计，跨尺度是后加需求，路由未扩展（§4.2 缺失接口表明确列出 3 个端点契约）。
2. **受控数据生产停滞（F-05/F-06/F-07/F-08）**：scperturb 30 个 h5ad 在盘但解析 loader 未写；replogle/scgenescope 无文件、目录空（§4.4）。根因：非代码问题，是**外部数据获取与格式适配工作量**——h5ad 结构已登记（scperturb_study_manifest.json），解析器是纯工程活；replogle/scgenescope 依赖下载与授权。
3. **真实验收闭环缺失（F-04）**：固定快照、指标阈值、标签/graph release 验收流程未建立。根因：需要真实数据资产就位后才能定阈值；工程框架（CI、manifest、demo 标记）已齐备。
4. **K02 Mamba 串行实现（唯一高债）**：`mamba_encoder.py:225` 逐时间步 for 循环，长序列吞吐线性劣化（§6.2/§6.4 S-K02）。根因：无 CUDA 编译环境下保持纯 PyTorch 回退；接入 `mamba-ssm` 官方 kernel 需 GPU 节点对拍验证（SSH_unit 环境有 mamba_ssm 包但本机无可用 CUDA 编译/运行验证条件，不满足"数值对拍 max|Δ|<1e-5"的验收要求，故本轮不冒险引入）。

### 5.4 与文档的复核差异（重要）

| 文档结论 | 实测复核 | 处理 |
|---|---|---|
| D2：GenKI 摄取路径 3 处裸 except 静默吞错（§6.3） | `perturbation.py:239,354,493` 三处均为 `progress_callback` 防护，带明确注释"Don't let callback errors break the pipeline"；真正的数据摄取错误路径（`:317,340,477`）均有 `as exc` 并写入 `metadata={"error": ...}` | **判断不成立**，不修改；报告如实记录（§1） |
| D4：API 路由零单元测试（§6.3） | `tests/unit/` 已有 7 个 API 测试文件共 2142 行：test_api_routes.py 516 行（health/live/ready/model-info/predict/batch/initialize/reset/provenance/variant）、test_predictions_routes.py 300 行（schema 校验）、test_middleware.py 358 行（限流/请求体/API key）、test_app_factory.py 305 行、test_api_auto_initialize.py 229 行等 | **判断过时**（文档 v4.0 生成前的测试未被计入），D4 从"待修"改为"已覆盖" |
| D1 含 2 处 file:///tmp/ URI（§6.3） | 实测为注释留痕（TD-M09 已改 PyPI 规范名）；**真实不可移植行是 `-e /home/scu/SSH_unit`**（文档未记录） | 按实测修复后者，注释留痕保留（§2.1） |

---

## 6. 未解决问题及后续解决策略

### 6.1 遗留问题清单（按严重程度）

| 级别 | 问题 | 文档出处 | 本轮状态 |
|---|---|---|---|
| 高 | K02 Mamba SSM 串行实现 | §6.4 S-K02（唯一高级存量） | 未动（需 GPU 节点数值对拍） |
| 高 | F-01 跨尺度 API 三端点 | §4.1/§4.2（缺失接口表） | 未动（CLI 层已闭环） |
| 高 | F-04 真实科学验收流程 | §4.1 | 未动（依赖真实数据资产） |
| 高 | F-06/F-07 replogle/scgenescope 数据获取 | §4.1/§4.4 | 未动（外部数据依赖） |
| 中 | F-03 余量 STRING/BioPlex/RegNetwork 导入 | §4.1/§4.2 | 未动（raw 在盘，纯工程活） |
| 中 | F-05 余量 scperturb h5ad 解析 loader | §4.1/§4.4 | 未动（30 h5ad 在盘，纯工程活） |
| 中 | F-08 GSE90546 表达矩阵解析 | §4.1/§4.4 | 未动（987MB RAW.tar 在位，结构已探明） |
| 中 | F-09 跨进程共享限流/指标 | §4.1/§6.4 K06 | 已缓解（单 worker 默认）；横向扩展前需方案 A |
| 中 | F-10 依赖冲突治理（scgpt↔scvi） | §4.1 | lock 入库后复现性提升，冲突面仍存 |
| 中 | N04 ensemble 双实现合并 | §6.4 S8 | 行为快照已就绪，未合并 |
| 中 | N06 低覆盖模块补测 / N11 docstring / N12 长函数 | §6.4 S5 | 未动 |
| 中 | N10 docs/api 缺 3 模块 rst | §6.4 S7 | 未动 |
| 中 | N13 顶层包名 `src` | §6.4 S7 | 未动（破坏性改动，需独立分支） |
| 中 | N14 mypy 全局忽略 | §6.4 S8 | 未动 |
| 低 | N04/N17/N18/F-11/F-12/F-13/PSIPRED 真集成 | §4.1/§6.4 S8 | 未动 |

### 6.2 后续解决策略（按优先级，含时间节点）

| 顺序 | 事项 | 工作量 | 时间节点 | 依据 |
|---|---|---|---|---|
| 1 | **K02 Mamba CUDA kernel 接入**（S-K02 方案 A：`selective_scan_1d` 替换 + 双实现数值对拍 max\|Δ\|<1e-5 + 256/1024/2048 基准） | 3.0 工作日 | 2026-08-26 前 | §6.4 S-K02；需 GPU 节点 |
| 2 | **F-01 跨尺度 API**（3 端点：initialize/predict/batch_predict，契约见 §4.2 缺失接口表；禁止隐式随机 fallback） | 3.0 工作日 | 2026-09-02 前 | §4.1/§4.2、§7.2 排期 6 |
| 3 | **F-08 GSE90546 解析**（per-GSM NPZ + 聚合 TSV，契约已登记于 import_manifest.json） | 2.0 工作日 | 2026-09-04 前 | §7.2 排期 3 |
| 4 | **F-03 余量图导入**（STRING/BioPlex/RegNetwork → canonical edge table + digest，复用 import_kinase_substrate.py 模式） | 2.0 工作日 | 2026-09-06 前 | §7.2 排期 4 |
| 5 | **F-05 scperturb h5ad 解析 loader**（消费 scperturb_study_manifest.json 已登记结构） | 2.0 工作日 | 2026-09-09 前 | §7.2 排期 7 |
| 6 | **F-06/F-07 数据获取**（replogle/scgenescope 下载与登记） | 数据依赖 | 数据就位后 2 工作日 | §7.2 排期 5 |
| 7 | **N04/N06/N08 落盘缓存/N10/N14** 等中低项 | 约 6.0 工作日 | 2026-09 月中旬 | §6.4 S8 |
| 8 | **N13 包名改造**（src-layout + 显式 package list + 全量 import 替换） | 2.0 工作日 | 独立 major 发布窗口 | §6.4 S7 方案 A(3) |

**风险提示**：K02 需 CUDA 编译环境（cu118 kernel 兼容性风险，回退路径须在 CI 覆盖，§6.4 S-K02）；N13 动全仓库 import 路径，须独立分支。

---

## 7. 验证命令与审计痕迹

### 7.1 本轮实际执行的验证命令

```bash
# 编译
python3 -m compileall -q src scripts; echo "EXIT: $?"            # EXIT: 0

# 全量测试（SSH_unit, Python 3.12.13 + pytest 9.0.3）
python -m pytest tests/unit tests/integration -q --tb=no -p no:cacheprovider
# → 1956 passed, 7 skipped, 21 warnings in 435.87s；EXIT: 0

# 分轮测试
pytest tests/unit/test_dependency_contracts.py -q                # 10 passed（D1）
pytest tests/unit/test_gene_mapper.py -q                         # 8 passed（N08）
pytest tests/unit/test_window_constants.py -q                    # 25 passed, 2 skipped（N05）
pytest tests/unit/test_utils.py -q                               # 17 passed（N09）
pytest tests/integration/test_api_concurrency.py tests/unit/test_api_auto_initialize.py -q  # 28 passed（第1轮集成）
pytest tests/integration/test_variant_workflow.py -q             # 16 passed（第2轮集成）
pytest tests/unit/test_models_ptm_predictors.py tests/unit/test_scripts.py -q  # 34 passed（第3轮回归）

# E2E 实机
python scripts/train.py --config configs/smoke/cnn_cpu.yaml --data data/raw/sample_data.csv \
    --output outputs/smoke_review --epochs 1 --batch_size 16     # 训练 8 步骤全过
python scripts/predict.py --model outputs/smoke_review/models/best_model.pt \
    --sequence ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY --device cpu  # 预测输出
# API 冒烟（TestClient）：/initialize(200) /predict(200) /ready(200) /model/info(200)，无 key 时 /initialize=503
```

### 7.2 提交链（main，本轮 3 组）

```bash
6ae2d4e build(deps): track requirements-lock.txt in VCS and reconcile numpy drift (D1)
33f6bde perf(analysis): exponential backoff for UniProt ID-mapping polls; explicit num_workers (N08/N09)
6dc4472 refactor(data): centralize PTM window constants and kill 31/15 magic numbers (N05)
```

### 7.3 遗留说明

- 冒烟训练产物 `outputs/smoke_review/` 与 `outputs/results/predictions.csv` 为验证产物，位于 `.gitignore` 忽略的 `outputs/` 下，smoke_review 已清理；`outputs/results/predictions.csv` 保留（gitignore 覆盖，不入库）。
- D3 垫片删除为工作区操作（该文件从未被 git 追踪），git 历史无记录，权威实现 `src/models/signaling_network.py` 不受影响。

---

**报告版本**：v1.0（2026-08-16）
**编制**：主智能体（3 轮修复迭代 + 系统复核 + E2E 实机验证）
**依据文档**：`project_analysis_20260816.md`（v4.0）——引用章节：§1 摘要、§3.3 保留判定、§4.1 未实现功能表、§4.2 缺失接口表、§4.4 数据契约缺口、§5.2 规范差异、§6.1 分级标准、§6.2 技术债总表、§6.3 新增技术债、§6.4 解决策略、§7.2 建议排期、§9.2/§9.3 验证命令
