# PTM2CellNet 项目综合分析报告（2026-08-16 v5.0）

> **任务依据**：`project_repair_report_20260816.md`（v1.0）"未解决问题及后续解决策略"章节（§6.1 遗留问题清单、§6.2 后续解决策略）为修复指引；接口契约引用本报告前版 v4.0（git `f5667db`，本文覆盖后可从 git 历史恢复）§4.1 未实现功能表、§4.2 缺失接口表。所有修复决策结合当前代码实测现状自主分析后独立做出。
> **执行方式**：3 轮完整"问题识别 → 方案设计 → 代码修改 → 单元测试 → 集成测试"迭代 + 1 项复核中发现遗留问题的补丁修复，随后系统性复核（E2E 训练/推理双链路实机验证）。
> **验证基线**：修复前 `main` = `6dc4472`（上轮 3 组修复提交，全量 1956 passed / 7 skipped）；修复后 `main` = `32fe202`（本轮 4 组提交）。全量测试 **2021 passed / 7 skipped / 0 failed**（441.52s，exit 0，净增 65 项），`compileall` exit 0，Sphinx docs 构建 exit 0。

---

## 目录

- [1. 问题修复汇总（按严重程度分类）](#1-问题修复汇总按严重程度分类)
- [2. 第 1 轮迭代：F-01 跨尺度 API 三端点](#2-第-1-轮迭代f-01-跨尺度-api-三端点)
- [3. 第 2 轮迭代：F-03 余量图数据导入（BioPlex/RegNetwork/STRING）](#3-第-2-轮迭代f-03-余量图数据导入bioplexregnetworkstring)
- [4. 第 3 轮迭代：N10 文档盲区 + N06 零测试模块补测](#4-第-3-轮迭代n10-文档盲区--n06-零测试模块补测)
- [5. 复核补丁：上轮 D3 垫片守卫测试遗留](#5-复核补丁上轮-d3-垫片守卫测试遗留)
- [6. 系统性复核：E2E 训练与推理技术要求评估](#6-系统性复核查-e2e-训练与推理技术要求评估)
- [7. 未解决问题及后续解决策略](#7-未解决问题及后续解决策略)
- [8. 验证命令与审计痕迹](#8-验证命令与审计痕迹)

---

## 1. 问题修复汇总（按严重程度分类）

严重程度沿用修复报告 §6.1 分级（高/中/低）。本轮修复依据其遗留清单选择在当前环境（无 GPU、无外网保证、受控数据未就位）下可完整闭环的项：**高级 1 项（F-01）、中级 4 项（F-03、N10、N06、上轮遗留测试债）**，并复核修正上轮报告 1 项判断。

| 严重程度 | 编号 | 问题（修复报告出处） | 修复状态 | 关键证据 |
|---|---|---|---|---|
| 高 | F-01 | 跨尺度模型 API 链路缺失（§6.1 第 2 位；v4.0 §4.1/§4.2 契约） | **已修复**（第 1 轮，`a63e4dd`） | `src/api/routes/cross_scale.py` 846 行；3 端点实机 200；API↔CLI 概率对拍 atol=1e-5 通过；34 单测 + 3 集成测试 |
| 中 | F-03 余量 | STRING/BioPlex/RegNetwork 原始图导入缺失（§6.1；§6.2 排期 4） | **主体修复**（第 2 轮，`aefd994`） | `scripts/import_ppi_graphs.py`；BioPlex 117,930 边 + RegNetwork 372,774 边 canonical 产物在盘；STRING 导入器就绪但 ENSP 映射文件待获取（数据获取项，见 §7） |
| 中 | N10 | docs/api 缺 `src.baselines`/`src.inference`/`src.project` 3 模块 rst（§6.1；v4.0 §6.4 S7） | **已修复**（第 3 轮，`d388efe`） | 3 个 rst + toctree；11 个顶层包全部有 autodoc 页面；`sphinx -b html` exit 0 |
| 中 | N06 | 零测试模块：`davf_velocity.py` 0 引用、`psipred.py` 别名未测（§6.1；v4.0 §6.4 S5） | **已修复**（第 3 轮，`d388efe`） | `test_davf_velocity.py` 14 测试（输出契约/确定性/梯度流/双注入/构造校验）+ PSIPREDClient 别名契约 3 测试 |
| 中（上轮遗留） | D3 尾巴 | 上轮删除根目录垫片后遗留 2 个守卫测试失败（全量 2 failed，见 §5） | **已修复**（`32fe202`） | 守卫对象已不存在，测试随垫片移除；全量恢复 0 failed |
| — | 复核修正 | 上轮 §6.2 排期 2 称 F-01 需 3.0 工作日 | **实测 1 轮闭环** | CLI 层契约（load_cross_scale_artifact 严格加载器）复用度高；仅图登记与状态槽为新增设计 |

**统计**：修复 5 项（高 1 + 中 4）、复核修正 1 项、新增测试 67 项（净增 65）、4 组提交（+2536 / −58 行，15 个文件）。

---

## 2. 第 1 轮迭代：F-01 跨尺度 API 三端点

### 2.1 问题识别

- **来源**：修复报告 §6.1（高，"F-01 跨尺度 API 三端点，CLI 层已闭环"）、§6.2 排期 2；接口契约 v4.0 §4.2：`/cross-scale/initialize`（artifact_path/device/strict_assets/max_batch_size）、`/cross-scale/predict`（sequence 或 embedding_ref 二选一、PTM sites）、`/cross-scale/batch_predict`（samples[]/batch_size/失败策略），并明确"禁止隐式随机 fallback"。
- **实测确认**：`grep -rn "cross_scale" src/api/` 0 命中；`initialize.py:214` 固定 `PTM2CellNet.from_config`；工程能力（`load_cross_scale_artifact` 严格加载器、`CrossScalePredictor`）全部就绪，缺的仅是 API 适配层——与修复报告 §5.3 根因分析一致。

### 2.2 方案设计（自主决策）

| 决策点 | 可选方案 | 决策与理由 |
|---|---|---|
| 状态管理 | A: 复用 `STATE`；B: 独立 `CROSS_SCALE_STATE` | **B**。两类模型输入/输出/生命周期契约完全不同（NPZ embedding 或序列 vs aa-index 张量）；标准 `initialize_model` 会重建 FeatureExtractor 并清除 variant workflow，混用互相破坏；独立槽位允许双服务并存 |
| 模型加载 | A: 仿标准 /initialize 手写加载；B: 复用 `load_cross_scale_artifact` | **B**。该加载器已具备 schema/label 词表/路径穿越/`weights_only=True`/strict 权重全部校验，重复实现徒增风险 |
| `strict_assets=False` 语义 | A: 整体放松校验；B: 仅放宽 best→last checkpoint 回退 | **B**。给 `load_cross_scale_artifact` 增加 `allow_last_checkpoint_fallback` 参数（最小扩展，其余校验严格如初） |
| cell graph 契约（实现中发现） | 模型 `CellGraphCompassHead` 硬性要求显式 `cell_edge_index`（"不可静默替换"），在线请求无法内嵌大图 | `initialize` 增加可选 `graph_ref`（服务端默认图 NPZ：cell_edge_index/signal_edge_index/signal_gene_map，含越界与 shape 校验）；单样本 NPZ 自带图优先；两者皆无 → 显式 400 |
| signal_gene_map 形状耦合（实现中发现） | 其 N_signal 维必须等于请求数据蛋白质节点数（序列路径 = 序列长度，随请求变化），静态默认图无法通吃 | 三级显式策略：NPZ/默认图自带（形状匹配时）> 请求显式 `allow_uniform_signal_map=true`（构造均匀映射 + `fallback_flags.signal_map_uniform` 标记，padding 节点零权重）> 显式 400 |
| PTM 类型映射 | A: 新建词表；B: 复用 `DEFAULT_PTM_TYPES` 单源（i+1 约定） | **B**。与 `PTMTokenAdapter`（1-based、0=padding）及标准 API `initialize_model` 约定一致 |
| batch 组批 | A: 全部逐样本；B: 纯序列样本组批 | **B**。模型 `forward` 原生支持序列列表（逐 residue 对齐）；带 PTM/embedding 样本逐样本（正确性优先） |

### 2.3 代码修改

| 文件 | 修改 |
|---|---|
| `src/api/routes/cross_scale.py`（新增，846 行） | 三端点 + `_CrossScaleState` + `reset_cross_scale_state` + 图/映射加载校验 + fallback 显式标记 |
| `src/api/routes/__init__.py` | 注册 cross_scale router（挂 `/api/v1` 前缀） |
| `src/inference/cross_scale_predictor.py` | `load_cross_scale_artifact` 增加 `allow_last_checkpoint_fallback` 参数（默认 False，行为不变） |
| `tests/unit/test_api_cross_scale.py`（新增） | 31 测试：initialize（digest/词表/摘要/门禁 401/403/503/穿越 403/404/last 回退/标准状态隔离/图校验）、predict（双输入互斥/PTM 映射与越界/embedding dim 校验/穿越/无图 400/无映射 400/NPZ 自带图/reset）、batch（组批/错误收集/fail_fast/空样本 422/未初始化 503）、词表单源契约 |
| `tests/integration/test_cross_scale_api.py`（新增） | 3 集成测试：真实 `train_cross_scale.py` 产物 → API 全链路；**API 与离线 CLI 概率数值一致性对拍（atol=1e-5）**；API key 门禁 |

### 2.4 测试结果

- **单元**：31 passed。首轮 8 个失败均为实现迭代中发现的真实契约问题（cell graph 必需、signal_gene_map 形状耦合、`Tensor.max(initial=)` 误用、线程池 HTTPException 透传），逐项修正后全绿。
- **集成**：3 passed（含 CLI 对拍）。回归：`test_api_routes` / `test_api_auto_initialize` / `test_api_concurrency` / 跨尺度全部单测与既有集成（pipeline/predict_cli/trainer/dataset/predictor）**115 passed**。
- **提交**：`a63e4dd`。

---

## 3. 第 2 轮迭代：F-03 余量图数据导入（BioPlex/RegNetwork/STRING）

### 3.1 问题识别

- **来源**：修复报告 §6.1（中，"raw 在盘，纯工程活"）、§6.2 排期 4（复用 import_kinase_substrate.py 模式）。
- **实测确认**：`datasets.yaml` 中 bioplex/regnetwork `loader: null`、string loader 指向消费方模块；`pathway_integration.py` 只接受预处理后边表。数据现状：BioPlex 117,930 边（基因符号）；RegNetwork 372,774 边（符号内嵌 + Entrez）；STRING 13.7M 全量 / 1.48M physical（ENSP 蛋白 ID）。
- **障碍实测**（自主探测）：① STRING 的 ENSP→symbol 映射文件（protein.info）不在盘，且 `data/raw/uniprot/human_idmapping.gz` 无 9606 前缀 STRING 条目（仅 4,593 条外源物种），离线映射不可行；② RegNetwork 方向表 `human.core.7z` sha256 与 SHA256SUMS.txt 一致（下载完整）但 py7zr/bsdtar 均报头错误（上游文件问题）。

### 3.2 方案设计（自主决策）

| 决策点 | 决策与理由 |
|---|---|
| 单脚本多源 vs 三脚本 | **单脚本 `--source {bioplex,regnetwork,string}`**：三源输出契约一致（统一 7 列 canonical 边表）、登记逻辑一致；分开则三份重复 boilerplate |
| 输出列 | 统一 `source_gene, target_gene, edge_type, score, evidence, species, release`（三源 canonical_schema required 并集；RegNetwork relation 编入 `edge_type=tf_regulation:{activates\|inhibits\|unspecified}` 词汇表，与 kinase_substrate `kinase_substrate:{modification}` 先例一致） |
| RegNetwork 7z 不可读 | **显式降级记录，不静默不丢弃**：relation=unspecified，manifest `relation_source` 字段记录 "unavailable (human.core.7z unreadable)"；core 文件就位后 `--regnetwork-core` 重跑即可补全（格式按官方三列假设并标注【假设】） |
| 符号缺失行 | human.node（Entrez→symbol）回填；仍无则保留 Entrez ID（与 kinase_substrate "未映射保留不丢弃" 同策略） |
| STRING 无映射文件 | 导入器完整实现（`--ensp-mapping` 接受 protein.info 风格 TSV，全 ID/裸 ENSP 双键兼容；`--min-score` 默认 700 对齐 datasets.yaml 注释）；未映射 ID 原样保留 + manifest 记录 mapping_rate。**正式产物待映射文件获取（数据获取项，非工程缺口）**，本轮以 `--max-rows 2000` 冒烟验证管道 |

### 3.3 代码修改

| 文件 | 修改 |
|---|---|
| `scripts/import_ppi_graphs.py`（新增，517 行） | 三源解析器（fail-fast 列头/行校验、path:line 错误）、统一边表输出、`ptm2cellnet.ppi-graph.import.v1` manifest（源 sha256/词表/参数） |
| `data/manifests/datasets.yaml` | 三源 `loader: null→scripts/import_ppi_graphs.py`、entrypoint 具体化（`--source <id>`）、preprocessing version 0.1.0→0.2.0、注释记录产物与遗留（STRING 映射、RegNetwork 7z） |
| `data/processed/{bioplex,regnetwork}/` | 产物在盘（`data/*` 按惯例 gitignore）：`bioplex_edges.tsv` 117,930 行、`regnetwork_edges.tsv` 372,774 行 + 各自 `import_manifest.json` |
| `tests/unit/scripts/test_import_ppi_graphs.py`（新增） | 14 测试：三源解析/契约违规/回填/过滤/映射双键/CLI roundtrip/manifest 契约 |

### 3.4 测试结果

- **单元**：14 passed。
- **集成回归**：`test_import_kinase_substrate` + `test_import_scperturb` + `test_data_manifest` + `test_data_manifest_cli` **83 passed**；`scripts/validate_data_manifest.py` exit 0。
- **实机产物**：BioPlex 117,930 / RegNetwork 372,774 边（均与官方发布数一致）；STRING 冒烟 2,000 边（mapping_rate=0 如实记录）。
- **提交**：`aefd994`。

---

## 4. 第 3 轮迭代：N10 文档盲区 + N06 零测试模块补测

### 4.1 问题识别

- **N10**（修复报告 §6.1；v4.0 §6.4 S7）：`docs/api/` 有 9 个 rst，缺 `src.baselines`（PMADS Ridge 基线）、`src.inference`（跨尺度推理加载器——本轮 F-01 的核心依赖）、`src.project`（范围注册表）3 个顶层包页面；11 个顶层包仅 8 个有 autodoc。
- **N06**（v4.0 §6.4 S5）：`src/models/davf_velocity.py`（167 行，DAVF Flow Matching 核心速度网络）**0 个测试引用**；`src/models/external_tools/psipred.py` 的 `PSIPREDClient` 弃用别名无契约测试（ChouFasmanClient 主体经聚合入口已有测试）。

### 4.2 方案设计

- N10：按现有 rst 模板（automodule + members/undoc-members/show-inheritance）补 3 页 + index toctree 追加；以 Sphinx 实际构建验证而非仅静态检查。
- N06：davf_velocity 为纯 nn.Module、完全可测——覆盖输出契约（形状/有限性/梯度流）、确定性（eval+dropout=0）、时间与条件敏感性、residual 门控初始化、hybrid/concat 双注入结构差异、构造期参数校验（非法 injection、modulation_dim 整除）。

### 4.3 代码修改

| 文件 | 修改 |
|---|---|
| `docs/api/{baselines,inference,project}.rst`（新增）+ `docs/api/index.rst` | 3 页 autodoc + toctree 补全 |
| `tests/unit/models/test_davf_velocity.py`（新增） | 14 测试（4 个测试类 + 参数化） |
| `tests/unit/test_external_tools.py` | 追加 `TestPSIPREDClientAlias` 3 测试（别名同一性/行为一致/内置可用性） |

### 4.4 测试结果

- **单元**：`test_davf_velocity.py` 14 passed；`test_external_tools.py` 45 passed（42 既有 + 3 新）；`tests/unit/models/` 全目录 57 passed。
- **Sphinx 构建**：`python3 -m sphinx -b html docs /tmp/docs_build` **exit 0**，`baselines.html`/`inference.html`/`project.html` 均生成且包含目标符号。
- **提交**：`d388efe`。

---

## 5. 复核补丁：上轮 D3 垫片守卫测试遗留

系统性复核首轮全量测试发现 **2 failed**：`tests/unit/test_signaling_network_import.py` 的 `test_root_signaling_network_imports_successfully` 与 `test_root_signaling_network_is_shim_of_canonical_impl`——二者硬性要求根目录 `signaling_network.py` 垫片可导入，而该垫片已在上轮 D3 修复中删除（修复报告 §4.1/§7.3）。上轮报告声称的 1956 passed 与垫片删除存在时序脱节（全量跑在删除前），守卫测试未同步移除。

处置：守卫对象已不存在，两个测试随垫片移除，文件头部补历史说明（D3 出处与权威实现位置）；其余 3 个导入卫生守卫（circular import / warning 检查）保留。修复后全量 **0 failed**。提交 `32fe202`。

---

## 6. 系统性复核：E2E 训练与推理技术要求评估

### 6.1 E2E 实机验证（本轮实际执行）

| 链路 | 命令/方式 | 结果 |
|---|---|---|
| 标准训练 | `python scripts/train.py --config configs/smoke/cnn_cpu.yaml --data data/raw/sample_data.csv --output outputs/smoke_review2 --epochs 1 --batch_size 16` | **通过**：完整训练 + best_val_loss 0.8079 + `artifact_manifest.json` 导出 + `model_kind=demo` 标记生效 |
| 标准推理 CLI | `python scripts/predict.py --model outputs/smoke_review2/models/best_model.pt --sequence ... --device cpu` | **通过**：apoptosis / 0.394 / 概率分布输出落盘 |
| 标准 API | TestClient：`/initialize`(200) → `/predict`(200, apoptosis 0.286) → `/ready`(200) → `/model/info`(200) | **通过** |
| 跨尺度训练 | `scripts/train_cross_scale.py`（合成 NPZ，允许 fallback 的显式 config） | **通过**（exit 0） |
| 跨尺度 API | `/cross-scale/initialize`(200, graph_registered=true) → `/predict` embedding(200, flags 全 false) → `/predict` sequence(200, uniform map 显式标记, delta_expression 4 维) → `/batch_predict`(200, 2/2) | **通过**（F-01 本轮闭环） |
| API↔CLI 数值对拍 | 同一 artifact + 同一 NPZ：API embedding 路径 vs `predict_cross_scale.py` | **一致**（概率 `np.allclose(atol=1e-5)`） |
| 图数据导入 | `import_ppi_graphs.py --source bioplex / regnetwork` | **通过**（117,930 + 372,774 边，manifest 完整） |
| docs 构建 | `sphinx -b html` | exit 0（11 包全覆盖） |
| 全量测试 | `pytest tests/unit tests/integration -q` | **2021 passed / 7 skipped / 0 failed，441.52s，exit 0**（基线 1956/7） |
| 编译 | `python3 -m compileall -q src scripts` | exit 0 |

已知非阻断观察：标准 `/initialize` 加载非 PTM-site checkpoint 时，variant workflow 内部 PTM-site predictor 逐类型尝试加载并告警（stderr 可见 state_dict 不匹配警告），最终 variant workflow 仍初始化成功、主预测链路 200——属既有告警性行为（对不匹配 checkpoint 显式报告而非静默），本轮未改变该语义。

### 6.2 满足度评估

| 技术要求 | 满足度 | 说明 |
|---|---|---|
| 标准训练 E2E（CNN/Transformer） | ✅ 满足 | 实机验证 + 全量测试 |
| 标准推理（CLI + API） | ✅ 满足 | 双链路实机 200 |
| 跨尺度训练/推理（CLI 层） | ✅ 满足 | 上轮闭环（修复报告 §5.2），本轮回归通过 |
| **跨尺度 API 服务（F-01）** | ✅ **满足（本轮闭环）** | 三端点实机验证 + CLI 数值对拍 + 门禁/路径安全/显式 fallback |
| 图数据导入（F-03） | ✅ 主体满足（本轮闭环 2/3 源） | BioPlex/RegNetwork canonical 产物在盘；STRING 待 ENSP 映射文件（数据获取项） |
| 文档覆盖（N10） | ✅ 满足（本轮闭环） | 11 顶层包 autodoc 全覆盖，构建通过 |
| 环境可复现性 | ✅ 满足（上轮闭环） | lock 入库（修复报告 §1 D1） |
| 真实科学验收流程（F-04） | ❌ 不满足 | `tests/real_assets/` 仍按门禁 skip；依赖真实数据资产定阈值 |
| 受控数据完整性 | ❌ 不满足 | scperturb h5ad 解析（F-05）、replogle/scgenescope 获取（F-06/F-07）、GSE90546 解析（F-08）未做 |
| 跨进程共享限流/指标（F-09） | ⚠️ 已缓解 | 单 worker 默认；横向扩展前需外部存储方案 |

**结论**：**E2E 训练与推理的全部工程链路（标准 + 跨尺度，CLI + API）已闭环且实机验证通过**。剩余缺口全部集中在两类：① 受控数据生产（F-05/F-06/F-07/F-08，外部数据获取与格式适配）；② 真实科学验收（F-04，依赖 ①）。二者均为数据/流程依赖而非代码工程缺口。

### 6.3 主要问题点与根本原因

1. **受控数据生产停滞（F-05/F-06/F-07/F-08）**：根因是外部数据获取（replogle/scgenescope 无文件；STRING ENSP 映射、RegNetwork core 均为上游资产问题）与大体量格式适配工作量（scperturb 21GB/30 h5ad、GSE90546 987MB tar）。工程框架（登记 manifest、导入器模板、fail-fast 契约）已就绪。
2. **真实科学验收闭环缺失（F-04）**：根因是需真实数据资产就位后才能固定快照与指标阈值；CI/manifest/demo 标记等工程框架齐备。
3. **K02 Mamba SSM 串行实现（唯一高债存量）**：`mamba_encoder.py:225` 逐时间步循环。根因：无 CUDA 编译/运行验证环境（本机 mamba_ssm 包存在但无法完成"双实现数值对拍 max|Δ|<1e-5"验收），维持修复报告 §5.3 第 4 条判断，不冒险引入。

---

## 7. 未解决问题及后续解决策略

### 7.1 遗留问题清单（按严重程度）

| 级别 | 问题 | 本轮状态 | 依据 |
|---|---|---|---|
| 高 | K02 Mamba SSM 串行实现 | 未动（需 GPU 节点对拍） | 修复报告 §6.1/§6.2 排期 1 |
| 高 | F-04 真实科学验收流程 | 未动（依赖真实数据资产） | 修复报告 §6.1 |
| 高 | F-06/F-07 replogle/scgenescope 数据获取 | 未动（外部数据依赖） | 修复报告 §6.1 |
| 中 | F-03 尾巴：STRING ENSP→symbol 映射文件获取 | **导入器就绪**，待 `--ensp-mapping` 文件（可由 STRING protein.info 构造）；RegNetwork core 7z 待上游可读版本或手动解压 | 本报告 §3 |
| 中 | F-05 scperturb h5ad 解析 loader | 未动（30 h5ad 在盘，结构 manifest 已登记） | 修复报告 §6.2 排期 5 |
| 中 | F-08 GSE90546 表达矩阵解析 | 未动（987MB RAW.tar 在位，解析契约已在 `import_norman_adamson.py:474-555` 定义） | 修复报告 §6.2 排期 3 |
| 中 | F-09 跨进程共享限流/指标 | 已缓解（单 worker 默认） | 修复报告 §6.1 |
| 中 | F-10 依赖冲突治理（scgpt↔scvi） | 未动 | 修复报告 §6.1 |
| 中 | N04 ensemble 双实现合并 | 未动（行为快照已就绪） | 修复报告 §6.1 |
| 中 | N06 余量：geneformer_embedding.py 无专属测试 / perturbation.py 44.6% 覆盖 | 部分缓解（davf_velocity + psipred 本轮闭环） | v4.0 §6.4 S5 |
| 中 | N11 docstring / N12 长函数（`_build_prediction_response` 实测已 374 行，较 v4.0 记录的 352 行继续膨胀） | 未动 | v4.0 §6.3 |
| 中 | N13 顶层包名 `src` / N14 mypy 收紧 | 未动（破坏性/全局改动） | 修复报告 §6.1 |
| 低 | F-11/F-12/F-13/PSIPRED 真集成 | 未动 | 修复报告 §6.1 |
| 低 | API 文档补录（`docs/guides/deployment.md` 未列跨尺度 3 端点） | **本轮新增待办**（v4.0 §4.2 文档补录项的增量） | 本报告 §2 |

### 7.2 后续解决策略（按优先级，含时间节点）

| 顺序 | 事项 | 工作量 | 时间节点 | 说明 |
|---|---|---|---|---|
| 1 | **K02 Mamba CUDA kernel 接入**（selective_scan_1d 替换 + 双实现对拍 max\|Δ\|<1e-5 + 256/1024/2048 基准） | 3.0 工作日 | 2026-08-26 前 | 修复报告 §6.2 排期 1；需 GPU 节点 |
| 2 | **F-08 GSE90546 解析**（per-GSM NPZ + 聚合 TSV，契约已编码于 `parse_gse90546`） | 2.0 工作日 | 2026-09-04 前 | 修复报告 §6.2 排期 3 |
| 3 | **F-05 scperturb h5ad 解析 loader**（消费 `scperturb_study_manifest.json` 已登记结构） | 2.0 工作日 | 2026-09-09 前 | 修复报告 §6.2 排期 5 |
| 4 | **F-03 尾巴**：下载 STRING protein.info 构造映射文件后重跑导入器；RegNetwork core 手动解压后 `--regnetwork-core` 重跑 | 0.5 工作日（含下载等待） | 2026-09-10 前 | 本报告 §3；纯数据获取 + 重跑 |
| 5 | **F-06/F-07 数据获取**（replogle/scgenescope 下载与登记） | 数据依赖 | 数据就位后 2 工作日 | 修复报告 §6.2 排期 6 |
| 6 | **F-04 真实科学验收流程**（固定快照/指标阈值/验收报告，依赖 2-5 数据就位） | 3.0 工作日 | 数据就位后 | 修复报告 §6.1 |
| 7 | **N04 合并 / N06 余量补测 / N11 / N12（predictions.py 长函数拆分）** | 约 5.0 工作日 | 2026-09 月中旬 | 修复报告 §6.2 排期 7 |
| 8 | **API 文档补录**：deployment.md 增补跨尺度 3 端点（initialize/predict/batch_predict）与 graph_ref 说明 | 0.5 工作日 | 随下次 docs 批次 | 本报告 §2 新增 |
| 9 | **N13 包名改造 / N14 mypy 收紧** | 2.0+ 工作日 | 独立 major 发布窗口 | 修复报告 §6.2 排期 8 |

**风险提示**：K02 需 CUDA 编译环境（cu118 kernel 兼容性，回退路径须 CI 覆盖）；F-03 尾巴的 STRING protein.info 下载依赖外网；N13 动全仓库 import 路径须独立分支。

---

## 8. 验证命令与审计痕迹

### 8.1 本轮实际执行的验证命令

```bash
# 编译与文档
python3 -m compileall -q src scripts; echo "EXIT: $?"          # EXIT: 0
(cd docs && python3 -m sphinx -b html -q . /tmp/docs_build)    # EXIT: 0（11 包页面全生成）

# 全量测试（SSH_unit, Python 3.12.13 + pytest 9.0.3）
python -m pytest tests/unit tests/integration -q --tb=no -p no:cacheprovider
# → 2021 passed, 7 skipped, 0 failed, 21 warnings in 441.52s；EXIT: 0（基线 1956/7）

# 分轮测试
pytest tests/unit/test_api_cross_scale.py -q                   # 31 passed（F-01）
pytest tests/integration/test_cross_scale_api.py -q            # 3 passed（含 CLI 对拍）
pytest tests/unit/scripts/test_import_ppi_graphs.py -q         # 14 passed（F-03）
pytest tests/unit/models/test_davf_velocity.py -q              # 14 passed（N06）
pytest tests/unit/test_external_tools.py -q                    # 45 passed（42+3）
pytest tests/unit/test_signaling_network_import.py tests/unit/test_middleware.py -q  # 28 passed（D3 尾巴）
pytest <API+跨尺度全量回归 12 文件> -q                          # 115 passed（第 1 轮回归）
pytest <导入器+manifest 5 文件> -q                             # 83 passed（第 2 轮回归）

# 图数据导入实机
python3 scripts/import_ppi_graphs.py --source bioplex     # 117,930 边
python3 scripts/import_ppi_graphs.py --source regnetwork  # 372,774 边
python3 scripts/import_ppi_graphs.py --source string --max-rows 2000 --output-root /tmp/string_smoke
python3 scripts/validate_data_manifest.py                 # EXIT: 0

# E2E 实机（标准链路）
python scripts/train.py --config configs/smoke/cnn_cpu.yaml --data data/raw/sample_data.csv \
    --output outputs/smoke_review2 --epochs 1 --batch_size 16
python scripts/predict.py --model outputs/smoke_review2/models/best_model.pt \
    --sequence ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY --device cpu
# API 冒烟（TestClient）：/initialize(200) /predict(200) /ready(200) /model/info(200)
# 跨尺度（TestClient）：训练 OK /initialize(200) /predict embedding(200) /predict sequence(200)
#   /batch_predict(200, 2/2)；API vs CLI 概率 allclose(atol=1e-5)
```

### 8.2 提交链（main，本轮 4 组）

```bash
a63e4dd feat(api): cross-scale online serving endpoints initialize/predict/batch_predict (F-01)
aefd994 feat(data): canonical graph importers for BioPlex/RegNetwork/STRING (F-03)
d388efe docs+test: cover remaining API modules and zero-test DAVF velocity field (N10/N06)
32fe202 test: drop obsolete root-shim guards removed with the D3 shim deletion
```

变更量：15 个文件，+2536 / −58 行。图数据 canonical 产物位于 `data/processed/{bioplex,regnetwork}/`（`data/*` 按项目惯例 gitignore，在盘可复核）。

### 8.3 遗留说明

- `outputs/smoke_review2/` 为本轮 E2E 验证产物（gitignore 覆盖，保留供复核，可删）。
- v4.0 分析报告被本版（v5.0）覆盖；v4.0 全文可从 git `f5667db` 恢复。
- RegNetwork `human.core.7z` 与 STRING 映射文件的获取状态已在 `datasets.yaml` 注释与各自 `import_manifest.json` 中显式登记。

---

**报告版本**：v5.0（2026-08-16）
**编制**：主智能体（3 轮修复迭代 + 1 项复核补丁 + 系统复核 + E2E 双链路实机验证）
**依据文档**：`project_repair_report_20260816.md`（v1.0）——引用章节：§1 修复汇总、§4.1/§4.3/§4.4（D1/N05/D3 详情）、§5.2 满足度、§5.3 根因、§5.4 复核差异、§6.1 遗留清单、§6.2 解决策略、§7 验证命令；`project_analysis_20260816.md`（v4.0，git `f5667db`）——引用章节：§4.1 未实现功能表、§4.2 缺失接口表（F-01 三端点契约）、§4.4 数据契约缺口、§6.3 新增技术债、§6.4 解决策略（S5/S7）、§7.2 排期
