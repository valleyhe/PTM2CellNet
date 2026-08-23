# PTM2CellNet 项目修复报告 — TD-BIO-01 / TD-LONG-01（2026-08-17）

> **任务依据**：最新分析文档 `project_analysis_20260816.md`（v5.0）§7"未解决问题及后续解决策略"章节提出的解决方案为修复指引；结合当前代码现状自主分析、独立决策最优修复方案。
> **执行方式**：3 轮完整"问题识别 → 方案设计 → 代码修改 → 单元测试 → 集成测试"迭代 + 1 项复核中发现遗留问题的补丁修复，随后 E2E 训练/推理双链路系统性复核（全部实机验证）。
> **验证基线**：修复前 `main` = `3da79c8`（在盘测试基线 2035 passed / 7 skipped，= v5.0 §8 记录的 2021 + 昨日 K02/F-05 会话新增 14 项）；修复后 `main` = `cea16f3`（本轮 4 组提交）。全量测试 **2064 passed / 7 skipped / 0 failed**（433.51s，exit 0，净增 29 项），`compileall` exit 0，Sphinx docs 构建 exit 0，`validate_data_manifest.py` exit 0。

---

## 0. 编号映射决策（重要前置说明）

任务指定的 **TD-BIO-01** 与 **TD-LONG-01** 两个编号，经全仓库检索（工作区全部文件、`git log --all -S` 全历史、`docs/archive/` 与根 `archive/` 三批归档）**均不存在**——v5.0 §7.1 遗留清单使用的是 K/F/N/D 系列编号。依据任务授权（"结合当前代码现状进行自主分析，独立决策最优修复方案"），按编号语义映射到 v5.0 §7 中语义最吻合且当前环境可完整闭环的未决项：

| 任务编号 | 映射到 | 依据 |
|---|---|---|
| **TD-LONG-01** | **N12 长函数治理**（v5.0 §7.1 表、§7.2 排期 7："`_build_prediction_response` 实测已 374 行，较 v4.0 记录的 352 行继续膨胀，未动"） | "LONG"= 长函数；v5.0 §7.1 中唯一的长函数治理项 |
| **TD-BIO-01** | **F-03 尾巴：STRING ENSP→symbol 生物学映射 + RegNetwork core**（v5.0 §7.1 表第 4 行、§7.2 排期 4："导入器就绪，待 `--ensp-mapping` 文件（可由 STRING protein.info 构造）；RegNetwork core 7z 待上游可读版本"） | "BIO"= 生物学数据链路；§7.1 中唯一纯生物学数据遗留且本轮环境（外网可达）可真实闭环 |

复核中修正了 v5.0 对这两个目标的三处**过时/错误记录**（详见 §1 与 §2/§3 各轮）。

---

## 目录

- [1. 问题修复汇总（按严重程度分类）](#1-问题修复汇总按严重程度分类)
- [2. 第 1 轮迭代：TD-LONG-01 — predictions.py 长函数治理（N12）](#2-第-1-轮迭代td-long-01--predictionspy-长函数治理n12)
- [3. 第 2 轮迭代：TD-BIO-01 — F-03 尾巴闭环（STRING 映射 + RegNetwork core）](#3-第-2-轮迭代td-bio-01--f-03-尾巴闭环string-映射--regnetwork-core)
- [4. 第 3 轮迭代：N06 余量补测 + API 文档补录](#4-第-3-轮迭代n06-余量补测--api-文档补录)
- [5. 复核补丁：GSE90546 守卫测试与可变在盘状态耦合](#5-复核补丁gse90546-守卫测试与可变在盘状态耦合)
- [6. 系统性复核：E2E 训练与推理技术要求评估](#6-系统性复核对-e2e-训练与推理技术要求评估)
- [7. 未解决问题及后续解决策略](#7-未解决问题及后续解决策略)
- [8. 验证命令与审计痕迹](#8-验证命令与审计痕迹)
- [9. 文档引用索引](#9-文档引用索引)

---

## 1. 问题修复汇总（按严重程度分类）

严重程度沿用 v5.0 §7.1 分级（高/中/低）。本轮 TD-BIO-01 在实施中升级出 1 项**高级生物学正确性缺陷**（现有 canonical 产物误标 199,578 条 miRNA 边）。

| 严重程度 | 编号 | 问题（文档出处） | 修复状态 | 关键证据 |
|---|---|---|---|---|
| 高（实施中发现） | TD-BIO-01 子项 | **RegNetwork canonical 产物生物学误标**：`human.source` 混含 193,115 条 hsa-miR* 调控边，导入器全部标注为 `tf_regulation:*`（v5.0 §3.3 产物） | **已修复**（第 2 轮，`565e29f`） | 重导后 `mirna_regulation:unspecified` 199,578 / `tf_regulation:unspecified` 171,593 / `regulation:unspecified` 1,600 / `lncrna_regulation:unspecified` 3；368,989 边（98.98%）由 core 真实定型 |
| 高 | TD-BIO-01 | STRING ENSP→symbol 映射缺失，正式 canonical 产物未生成（v5.0 §7.1 第 4 行、§3.2） | **已修复**（第 2 轮，`565e29f`） | `9606.protein.info.v12.0.txt.gz` 官方下载（sha256 入 SUMS）→ 派生 19,699 蛋白两列映射 → physical 全量导入 **173,038 边 / 10,746 节点 / mapping_rate=1.0 / 0 未映射** |
| 高 | TD-BIO-01 子项 | RegNetwork `human.core.7z` 不可读，v5.0 §3.2 判为"上游文件问题" | **已修复且修正根因**（第 2 轮） | 实为**本地下载截断**（在盘 1,248,947 字节 vs 头部 NextHeaderOffset 0x2fe69a≈3.14MB）；从 zpliulab.cn 官方路径重下完整 3,139,380 字节，py7zr 解压成功（811,766 行） |
| 中 | TD-LONG-01 | N12 长函数膨胀（v5.0 §7.1；文档记录 `_build_prediction_response` 374 行） | **已修复（标的修正）**（第 1 轮，`a0cfb61`） | 文档 374 行为"函数起点到文件尾"误测（419→792=373）；实际最长函数 `predict_variant` 179 行 → 拆分后主体 83 行 + 5 个内聚辅助函数（12/19/4/50/47 行） |
| 中 | TD-LONG-01 附带 | 导入器 CRLF 处理缺陷：`rstrip("\n")` 不去 `\r`，CRLF 源文件的 Entrez 键带 `\r` 残留，静默破坏回填/join | **已修复**（第 2 轮，`565e29f`） | 全部 6 处解析循环改 `rstrip("\r\n")`；回归测试锁定（`test_crlf_source_does_not_leak_or_break_backfill`） |
| 中 | N06 余量 | `geneformer_embedding.py` 无专属测试（v5.0 §7.1 倒数第 3 行；v4.0 §6.4 S5） | **已修复**（第 3 轮，`85998f7`） | `test_geneformer_embedding.py` 13 测试（fallback 警告语义/strict 门禁+环境覆盖/稳定哈希/整数与越界查找/设备迁移/单例助手），离线安全 |
| 中 | API 文档补录 | `docs/guides/deployment.md` 未列跨尺度 3 端点（v5.0 §7.1 末行、§7.2 排期 8） | **已修复**（第 3 轮，`85998f7`） | 新增"跨尺度推理端点"整节（端点表/`graph_ref` 契约/`allow_uniform_signal_map` 显式降级/批量语义），并与代码逐条核对（初稿 422 经核对代码订正为 400） |
| 中（复核补丁） | 上轮遗留 | `test_data_contract_fixes.py` GSE90546 守卫断言在盘报告 `status=="probed"`，被昨日 F-08 真实解析合法推进为 `parsed` 后击穿（全量 1 failed） | **已修复**（`cea16f3`） | 守卫改为直接调用 `probe_gse90546()` 断言返回契约，与在盘可变状态解耦；全量恢复 0 failed |

**统计**：4 组提交（`a0cfb61` / `565e29f` / `85998f7` / `cea16f3`），8 个文件 +718/−200 行，新增测试 29 项（12 特征化 + 4 导入器净增 + 13 geneformer），真实数据产物：STRING 173,038 边 + RegNetwork 重标 372,774 边 + core 811,766 行在盘。

---

## 2. 第 1 轮迭代：TD-LONG-01 — predictions.py 长函数治理（N12）

### 2.1 问题识别

- **来源**：v5.0 §7.1（中，"N12 长函数（`_build_prediction_response` 实测已 374 行，较 v4.0 记录的 352 行继续膨胀）| 未动"）、§7.2 排期 7。
- **实测复核（与文档不符）**：AST 精确测量 `src/api/routes/predictions.py`，`_build_prediction_response` 实为 **26 行**（`predictions.py:419-444`）——v4.0 记 352 行、v5.0 记 374 行，两版均为"函数 def 起点到文件尾（792 行）"的误测。文件真正的问题函数是 **`predict_variant`（179 行，`predictions.py:611-789`）**。全仓扫描另确认 35 个 ≥100 行函数中本文件占 3 席。
- **测试安全网评估**：variant 端点仅 2 个用例（503 未初始化 + mock happy path），不足以支撑行为保持重构。

### 2.2 方案设计（自主决策）

| 决策点 | 决策与理由 |
|---|---|
| 修复标的 | `predict_variant`（真实最长、六段异质逻辑混杂），而非文档点名的 `_build_prediction_response`（仅 26 行，无需拆） |
| 测试先行 | 先补 12 个特征化测试锁死行为契约再动代码（错误映射 504/400×6 参数化、缺序列 400、置信度公式、cell-state 成功/降级、pathway 置信度阈值） |
| 拆分粒度 | 按内聚职责提 5 个模块级辅助函数：`_ptm_effects_to_models` / `_pathway_impacts_to_models` / `_variant_confidence` / `_resolve_variant_sequence`（async，UniProt 三类异常映射）/ `_variant_cell_state_prediction`（async，模型失败降级为 warning 不 500） |
| 不拆的部分 | 端点级 try/except 错误策略保留在编排主体（对外契约可见性优先）；`batch_preprocess`（111 行）为线性向量化代码、单一职责，不为行数指标做外观拆分 |

### 2.3 代码修改

| 文件 | 修改 |
|---|---|
| `src/api/routes/predictions.py` | `predict_variant` 179→83 行；新增 5 辅助函数；行为逐分支保持 |
| `tests/unit/test_api_routes.py` | `TestVariantPredictRoute` 追加 12 特征化测试（含 6 路参数化错误映射） |

### 2.4 测试结果

- **单元**：`test_api_routes.py` **44 passed**（32 既有 + 12 新增；特征化测试对重构前实现先验证通过后动代码）。
- **集成**：`test_api.py` + `test_api_concurrency.py` + `test_variant_workflow.py`(集成) + `test_variant_effect.py` **56 passed**。
- **提交**：`a0cfb61`。

### 2.5 未解决问题

- `batch_preprocess`（111 行）保持现状（线性向量化，见 §2.2 决策理由）；全仓其余 ≥100 行函数（`self_supervised.pretrain_masked_ptm` 284 行等）属 v5.0 §7.2 排期 7 的 N12 余量，未在本轮 TD-LONG-01 范围。

---

## 3. 第 2 轮迭代：TD-BIO-01 — F-03 尾巴闭环（STRING 映射 + RegNetwork core）

### 3.1 问题识别

- **来源**：v5.0 §7.1 第 4 行（中）、§7.2 排期 4（"下载 STRING protein.info 构造映射文件后重跑导入器；RegNetwork core 手动解压后 `--regnetwork-core` 重跑"）、§3.2 障碍记录。
- **实测复核**：
  1. STRING 导入器（`scripts/import_ppi_graphs.py`）就绪但仅有 2,000 行冒烟产物（mapping_rate=0）；v5.0 §3.2 称离线映射不可行——本轮实测**外网可达**（string-db.org 200），障碍不成立。
  2. RegNetwork `human.core.7z`：7z 魔数合法，但头部声明 NextHeaderOffset≈3.14MB 而文件仅 1.25MB——**下载截断**；SHA256SUMS.txt 与文件一致只能证明"记录的是截断件"（该 SUMS 为下载时本地生成），v5.0 §3.2"上游文件问题"判断**有误**。
  3. 官网 regnetworkweb.org 503 维护中，但 zpliulab.cn 的 `/RegNetwork/static/data/human.core.7z` 路径可用。

### 3.2 方案设计（自主决策）

| 决策点 | 决策与理由 |
|---|---|
| STRING 映射来源 | 官方 `9606.protein.info.v12.0.txt.gz`（与在盘 links 同版本）→ 派生两列 canonical TSV（`9606.ENSP…→symbol`，19,699 行），raw 原件与派生件 sha256 均入 SUMS 与 datasets.yaml |
| 7z 重下 | 从 zpliulab.cn 官方静态路径重下（3,139,380 字节，与头部 offset 吻合），py7zr 解压 |
| core 消费契约 | 解压实测 core 为 **6 列无方向符号表**（regulator_symbol/id, target_symbol/entrez, 双侧类型）——v5.0 §3.2 的"3 列带 sign（±/?）"为错误【假设】。**RegNetwork 批量文件不发布 activation/inhibition，不捏造方向**；core 的真实价值 = 按 `(regulator_id, target_entrez)` join 提供调控者生物学类型（实测覆盖率 **98.98%**，368,989/372,774） |
| edge_type 词汇 | core 命中 → `{tf\|mirna\|lncrna\|circrna}_regulation:unspecified`；未命中 → `hsa-miR-*` 命名规约归 mirna、其余 neutral `regulation:unspecified`（不冒充 TF）。**此举顺带修复高级生物学缺陷**：旧产物把全部 372,774 边（含 199,578 条 miRNA 边）标为 `tf_regulation:*` |
| CRLF | `human.source` 官方为 CRLF 而解析用 `rstrip("\n")`——`\r` 残留会静默破坏 Entrez 回填与 core join；全部解析循环改 `rstrip("\r\n")` 并加回归测试 |
| core 键缺失行 | 811,766 行实测 1 行 circRNA 目标无 Entrez：跳过并计数进 manifest（显式，不静默，不为 1 行放弃 81 万行标注） |

### 3.3 代码修改

| 文件 | 修改 |
|---|---|
| `scripts/import_ppi_graphs.py` | docstring 更正为 6 列真实契约；`load_regnetwork_core` 重写（返回 `(id,id)->type` 映射 + 键缺失计数）；新增 `_regnetwork_edge_type`；词表换 `REGNETWORK_REGULATOR_TO_EDGE_PREFIX`（删除基于错误假设的 ±/? 词表）；6 处 CRLF 修复；CLI help 与 summary 键更新 |
| `tests/unit/scripts/test_import_ppi_graphs.py` | fixture 改真实格式（CRLF source + 6 列 core）；RegNetwork 测试类重写 + 3 新增（CRLF 回归、core 格式 fail-fast、未知 regulator_type fail-fast、键缺失跳过计数） |
| `data/manifests/datasets.yaml` | string 段：+2 文件登记（protein.info/派生映射）、entrypoint 带映射参数、preprocessing 0.2.0→0.3.0、注释改闭环事实；regnetwork 段：+human.core.txt 登记、0.3.0、注释记录截断根因与生物学重标 |
| `data/raw/{string,regnetwork}/SHA256SUMS.txt` | 新文件 sha256 入册（regnetwork 的 7z 条目更新为完整件） |
| `data/raw/string/`、`data/raw/regnetwork/` | protein.info 原件、派生映射 TSV、完整 7z、解压 human.core.txt 落位 |
| `data/processed/{string,regnetwork}/` | 真实产物：`string_edges.tsv` 173,038 行（mapping_rate 1.0）、`regnetwork_edges.tsv` 372,774 行重标 + 各自 `import_manifest.json` |

### 3.4 测试结果

- **单元**：`test_import_ppi_graphs.py` **18 passed**（14→18）。首跑真实 core 曾触发 fail-fast（第 804,781 行 circRNA 目标 Entrez 为空）——按 §3.2 设计处置后通过。
- **集成**：导入器 + kinase_substrate + scperturb + data_manifest（单测+CLI）共 **63 passed**；`validate_data_manifest.py` exit 0（ok=true）。
- **实机产物核验**：STRING 173,038 边全部为基因符号（抽查 M6PR/PLIN3/IGF2R 等）；RegNetwork edge_type 分布与 join 统计完全对账（199,578+171,593+1,600+3=372,774；core 定型 368,989 + 规约 2,185 + neutral 1,600）。
- **提交**：`565e29f`。

### 3.5 未解决问题

- STRING **全量变体**（`--string-full`，13.7M 边）未导入——datasets.yaml 契约默认 physical 子网（min_score≥700），全量为可选路径，非本轮范围（v5.0 §3 亦以 physical 为正式产物）。
- RegNetwork 方向（activates/inhibits）**数据源本身不提供**，属上游能力边界而非工程缺口，已在 manifest `typing_source` 显式记录。

---

## 4. 第 3 轮迭代：N06 余量补测 + API 文档补录

### 4.1 问题识别

- **N06 余量**（v5.0 §7.1 倒数第 3 行；v4.0 §6.4 S5）：`src/models/geneformer_embedding.py`（DAVF 的基因嵌入底座）无专属测试文件，仅被 test_davf 间接覆盖。
- **API 文档补录**（v5.0 §7.1 末行、§7.2 排期 8）：`docs/guides/deployment.md` 171 行中无任何跨尺度端点内容（grep 0 命中），而 F-01 三端点已上线两轮。
- 附带复核：AGENTS.md 技术债清单所列"长序列窗口逻辑重复"——实测 `SlidingWindowHandler._create_windows` 已委托 `LongSequenceHandler`（`long_sequence.py:340-355`，前轮 `6dc4472` 完成），该债**已过时**；顺带手工推演 `_compute_window_boundaries` 的尾部覆盖/整窗/小于窗三类边界均正确，无需改动。

### 4.2 方案设计

- N06：不触网的契约测试（model_path 指向不存在的本地绝对路径 + `HF_HUB_OFFLINE=1` 兜底，避免 CI 误连 hub）；覆盖 fallback 显式警告（"NOT suitable for production"）、strict 双通道（构造参数 + `PTM2CELLNET_STRICT_MODEL_ASSETS` 环境覆盖）、稳定哈希映射（确定性/词表界内/与查找一致）、整数索引直查与越界告警兜底 0、`embeddings` 未加载即 RuntimeError、`to()` 设备迁移、模块级单例（缓存/路径变更重建/provenance 三态）。
- 文档补录：按 `src/api/routes/cross_scale.py` 实际契约撰写（`_validate_single_input` 400、空 samples 422、`allow_uniform_signal_map` 默认 false 缺映射显式 400、`fail_fast` 语义、`graph_ref` 数组与越界校验），并经与代码/测试逐条核对后定稿。

### 4.3 代码修改

| 文件 | 修改 |
|---|---|
| `tests/unit/models/test_geneformer_embedding.py`（新增） | 13 测试 / 3 测试类 |
| `docs/guides/deployment.md` | 新增"跨尺度推理端点（cross-scale）"整节（5 小节），并恢复被误吞的"## 监控"标题（本轮自查发现并修复） |

### 4.4 测试结果

- **单元**：`test_geneformer_embedding.py` **13 passed**。
- **集成**：`test_cross_scale_api.py`(3，含 API↔CLI 数值对拍 atol=1e-5) + `test_api_cross_scale.py`(31) + `test_davf.py`(16) 共 **50 passed**。
- **文档构建**：`sphinx -b html` **exit 0**（仅 3 条既存归档文档 myst.xref 警告，与本轮改动无关）。
- **提交**：`85998f7`。

### 4.5 未解决问题

- v4.0 §6.4 S5 的另一项"`perturbation.py` 44.6% 覆盖"未动（覆盖提升型任务，非契约缺失）。

---

## 5. 复核补丁：GSE90546 守卫测试与可变在盘状态耦合

第 3 轮后全量回归首轮发现 **1 failed**：`tests/integration/test_data_contract_fixes.py::TestGSE90546StatusSemantics::test_probe_status_is_probed_not_parsed` 断言在盘 `GSE90546_structure_report.json` 的 `status=="probed"`，而昨日 F-08 会话（`project_repair_report_20260817_k02_f08_f05.md` §3）已真实执行 `--parse-gse90546` 将在盘状态合法推进为 `parsed`——**测试对可变在盘产物的硬编码耦合**，与 v5.0 §5 记录的 D3 垫片守卫属同类问题（上轮报告声称的通过状态与后续状态推进存在时序脱节）。

处置：不删除覆盖，改为**直接调用 `probe_gse90546()`**（真实 RAW.tar，只读）断言其返回 `status=="probed"` 的函数契约；tar 缺失时按文件惯例 skip；顺带删除失去消费者的 `GSE90546_REPORT` 死常量。修复后全量 **2064 passed / 7 skipped / 0 failed**。提交 `cea16f3`。

---

## 6. 系统性复核：E2E 训练与推理技术要求评估

### 6.1 E2E 实机验证（本轮实际执行）

| 链路 | 命令/方式 | 结果 |
|---|---|---|
| 标准训练 | `python scripts/train.py --config configs/smoke/cnn_cpu.yaml --data data/raw/sample_data.csv --output outputs/smoke_repair_20260817 --epochs 1 --batch_size 16` | **通过**（exit 0）：完整训练 + best_val_loss 0.8079 + `artifact_manifest.json` 导出 + `model_kind=demo` 显式标记 |
| 标准推理 CLI | `python scripts/predict.py --model outputs/smoke_repair_20260817/models/best_model.pt --sequence ACDE…WY --device cpu` | **通过**（exit 0）：apoptosis / 0.394 / 概率分布落盘 |
| 标准 API | TestClient（`PTM2CELLNET_ENV=development`）：`/initialize`(200) → `/predict`(200, apoptosis 0.286, model_kind=demo) → `/ready`(200) → `/model/info`(200) | **通过**；生产模式下 `/initialize` 无 API key 被门禁拒绝（503 + 明确 detail），安全语义符合预期 |
| 跨尺度训练/推理/API | 最终 HEAD 全量套内 `tests/integration/test_cross_scale_api.py` 3 项（真实 `train_cross_scale.py` 产物 → API 三端点 → API↔CLI 概率对拍 atol=1e-5） | **通过**（含在 2064 passed 中） |
| 图数据导入（真实数据） | STRING physical 全量 173,038 边（mapping_rate=1.0）；RegNetwork 372,774 边重标（core 定型 98.98%） | **通过**；`validate_data_manifest.py` exit 0 |
| 全量测试 | `pytest tests/unit tests/integration -q` | **2064 passed / 7 skipped / 0 failed，433.51s，exit 0**（会话起点 2035/7，净增 29） |
| 编译 / 文档 | `compileall src scripts` / `sphinx -b html` | 均exit 0 |

已知非阻断观察（与 v5.0 §6.1 一致，未改变语义）：标准 `/initialize` 加载非 PTM-site checkpoint 时 variant workflow 内部逐类型尝试加载并告警，最终仍初始化成功；API 重复打印两次启动日志（create_app 双次调用所致，仅日志噪音）。

### 6.2 满足度评估

| 技术要求 | 满足度 | 说明 |
|---|---|---|
| 标准训练/推理（CLI + API） | ✅ 满足 | 本轮实机复验（§6.1 前三行） |
| 跨尺度训练/推理（CLI + API + 数值对拍） | ✅ 满足 | 最终 HEAD 集成测试覆盖；文档补录闭环（§4） |
| Mamba 长序列性能（K02） | ✅ 满足（昨日闭环） | `3da79c8` 三路径对拍 + 基准（fused 7.07× at L=2048），本轮全量回归通过 |
| 图数据导入（F-03 全量） | ✅ **满足（本轮闭环最后一环）** | BioPlex（117,930）+ RegNetwork（372,774，带真实生物学类型）+ STRING（173,038，mapping 1.0）三源 canonical 产物全部在盘且 manifest 完整 |
| 受控数据解析（F-05/F-08） | ✅ 满足（昨日闭环） | scperturb h5ad 端到端 loader + GSE90546 86K cells/316M nnz |
| 文档覆盖（N10 + API 补录） | ✅ 满足 | 11 包 autodoc + deployment 跨尺度节 |
| 环境可复现性 | ✅ 满足 | lock 入库（v5.0 §6.2 D1） |
| 真实科学验收流程（F-04） | ❌ 不满足 | 依赖真实数据资产定阈值；`tests/real_assets/` 仍按门禁 skip |
| 受控数据完整性（F-06/F-07） | ❌ 不满足 | `data/raw/replogle`、`data/raw/scgenescope` 不在盘（本轮实测确认），外部获取依赖 |
| 跨进程共享限流/指标（F-09） | ⚠️ 已缓解 | 单 worker 默认；横向扩展前需外部存储 |

**结论**：**E2E 训练与推理的全部工程链路（标准 + 跨尺度、CLI + API、图数据导入、文档构建、全量测试）在最终 HEAD 实机验证通过且 0 failed**。剩余缺口集中在两类：① 外部数据获取（F-06/F-07）；② 依赖真实数据的科学验收（F-04）。二者为数据/流程依赖而非代码工程缺口——与 v5.0 §6.2 结论一致，且本轮把其中原判"上游损坏"的 F-03 尾巴证明为可工程闭环并已闭环。

### 6.3 主要问题点与根本原因（剩余项）

1. **F-04 真实科学验收未闭环**：根因是需真实数据资产就位后固定快照与指标阈值；CI/manifest/demo 标记等工程框架齐备（v5.0 §6.3 第 2 条判断维持）。
2. **F-06/F-07 数据不在盘**：replogle/scgenescope 为外部大文件下载（figshare/受控源），本轮网络虽通但属大体量数据获取任务，超出本轮 TD 编号范围。
3. **测试债的模式性问题**：本轮再次出现"守卫测试耦合可变在盘状态"（§5，与 v5.0 §5 D3 同类）。根因是部分集成测试以在盘产物为固定断言对象，而数据会话（F-08 等）会合法推进在盘状态。建议后续新增守卫测试一律以**函数返回契约**或 tmp_path 合成产物为断言对象。

---

## 7. 未解决问题及后续解决策略

### 7.1 遗留问题清单（按严重程度）

| 级别 | 问题 | 本轮状态 | 依据 |
|---|---|---|---|
| 高 | F-04 真实科学验收流程 | 未动（依赖真实数据资产） | v5.0 §6.1/§7.1 |
| 高 | F-06/F-07 replogle/scgenescope 数据获取 | 未动（`data/raw/` 实测不在盘） | v5.0 §7.1；k02_f08_f05 报告 §5.2 |
| 中 | F-09 跨进程共享限流/指标 | 已缓解（单 worker 默认） | v5.0 §7.1 |
| 中 | F-10 依赖冲突治理（scgpt↔scvi） | 未动 | v5.0 §7.1 |
| 中 | N04 ensemble 双实现合并 | 未动（行为快照已就绪） | v5.0 §7.1 |
| 中 | N11 docstring / N12 余量（全仓其他长函数，如 `pretrain_masked_ptm` 284 行、`perturbation.py` 44.6% 覆盖） | 部分缓解（本轮闭环 predictions.py 与 geneformer 测试） | v5.0 §7.1；v4.0 §6.4 S5 |
| 中 | N13 顶层包名 `src` / N14 mypy 收紧 | 未动（破坏性/全局改动） | v5.0 §7.1 |
| 低 | F-11/F-12/F-13/PSIPRED 真集成 | 未动 | v5.0 §7.1 |
| 低 | STRING 全量变体（13.7M 边）导入 | 可选路径未执行（契约默认 physical） | 本报告 §3.5 |
| 低 | API 启动日志双次打印（create_app 双调用） | 未动（纯日志噪音） | 本报告 §6.1 |

### 7.2 后续解决策略（按优先级，含时间节点）

| 顺序 | 事项 | 工作量 | 时间节点 | 说明 |
|---|---|---|---|---|
| 1 | **F-06/F-07 数据获取与登记**（replogle Pert-seq 矩阵、scgenescope 资产；下载→SHA256→datasets.yaml→导入器或 manifest 登记） | 数据依赖，就位后 2.0 工作日 | 2026-08-28 前 | 本轮实测外网可达，下载通道已验证（STRING/zpliulab 两源成功），建议优先推进 |
| 2 | **F-04 真实科学验收流程**（固定快照/指标阈值/验收报告；依赖顺序 1 与既有 F-05/F-08 受控数据） | 3.0 工作日 | 数据就位后 | v5.0 §7.2 排期 6 |
| 3 | **守卫测试去在盘耦合专项**：排查其余以 `data/processed/*` 为固定断言的集成测试，改为函数契约/tmp_path 断言 | 1.0 工作日 | 2026-09-05 前 | 本报告 §6.3 第 3 条（D3 + GSE90546 两次同类教训） |
| 4 | **F-09 跨进程限流/指标**（Redis/网关方案） | 2.0 工作日 | 2026-09-12 前 | v5.0 §7.2 排期 5 |
| 5 | **N04 合并 / N11 / N12 余量（全仓长函数与 docstring）/ perturbation.py 覆盖** | 约 5.0 工作日 | 2026-09 月中旬 | v5.0 §7.2 排期 7 |
| 6 | **N13 包名改造 / N14 mypy 收紧 / F-10 依赖治理** | 2.0+ 工作日 | 独立 major 发布窗口 | v5.0 §7.2 排期 8 |
| 7 | （可选）STRING 全量变体导入 `--string-full` | 0.5 工作日 | 按需 | 下游需要全量置信度网络时执行 |

**风险提示**：F-06/F-07 的下载体积大（RepLogle 全矩阵数 GB），需评估磁盘与断点续传；顺序 3 属测试稳健性投资，成本低但可消除数据会话与测试会话的相互击穿模式。

---

## 8. 验证命令与审计痕迹

### 8.1 本轮实际执行的验证命令

```bash
# 编译 / 文档 / manifest
python3 -m compileall -q src scripts                         # EXIT: 0
(cd docs && python3 -m sphinx -b html -q . /tmp/docs_build)  # EXIT: 0
python3 scripts/validate_data_manifest.py                    # EXIT: 0 (ok=true)

# 全量测试（SSH_unit, Python 3.12.13 + pytest 9.0.3；最终 HEAD）
python -m pytest tests/unit tests/integration -q --tb=no -p no:cacheprovider
# → 2064 passed, 7 skipped, 0 failed in 433.51s；EXIT: 0（会话起点 2035/7）

# 分轮测试
pytest tests/unit/test_api_routes.py -q                                    # 44 passed（TD-LONG-01）
pytest tests/unit/test_api.py tests/integration/test_api_concurrency.py \
        tests/integration/test_variant_workflow.py tests/unit/test_variant_effect.py -q  # 56 passed
pytest tests/unit/scripts/test_import_ppi_graphs.py -q                     # 18 passed（TD-BIO-01）
pytest <导入器+manifest 5 文件> -q                                          # 63 passed
pytest tests/unit/models/test_geneformer_embedding.py -q                   # 13 passed（N06 余量）
pytest tests/integration/test_cross_scale_api.py tests/unit/test_api_cross_scale.py \
        tests/unit/test_davf.py -q                                         # 50 passed（第 3 轮）
pytest tests/integration/test_data_contract_fixes.py -q                    # 6 passed（复核补丁）

# 图数据导入实机（真实数据）
python3 scripts/import_ppi_graphs.py --source string \
    --ensp-mapping data/raw/string/9606.protein_to_symbol.v12.0.tsv
#   → 173,038 边 / 10,746 节点 / n_unmapped=0 / mapping_rate=1.0
python3 scripts/import_ppi_graphs.py --source regnetwork \
    --regnetwork-core data/raw/regnetwork/human.core.txt
#   → 372,774 边 / n_typed_by_core=368,989 / mirna 199,578 + tf 171,593 + neutral 1,600 + lncrna 3

# E2E 实机（标准链路，最终 HEAD）
python scripts/train.py --config configs/smoke/cnn_cpu.yaml --data data/raw/sample_data.csv \
    --output outputs/smoke_repair_20260817 --epochs 1 --batch_size 16      # exit 0, best_val_loss 0.8079
python scripts/predict.py --model outputs/smoke_repair_20260817/models/best_model.pt \
    --sequence ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY --device cpu       # exit 0, apoptosis 0.394
# API 冒烟（TestClient, PTM2CELLNET_ENV=development）：
#   /initialize(200) /predict(200, apoptosis 0.286) /ready(200) /model/info(200)
```

### 8.2 提交链（main，本轮 4 组）

```bash
a0cfb61 refactor(api): decompose predict_variant into cohesive helpers (TD-LONG-01/N12)
565e29f feat(data): close F-03 tail — real STRING ENSP mapping + RegNetwork core biological typing (TD-BIO-01)
85998f7 test+docs: geneformer embedding contract suite and cross-scale deployment guide (N06 remainder / API doc backfill)
cea16f3 test: decouple GSE90546 probe-semantics guard from mutable on-disk report
```

变更量：8 个文件，+718 / −200 行。数据产物位于 `data/processed/{string,regnetwork}/`、新 raw 资产位于 `data/raw/{string,regnetwork}/`（`data/*` 按项目惯例 gitignore，在盘可复核，sha256 已入各自 SHA256SUMS.txt 与 datasets.yaml）。

### 8.3 遗留说明

- `outputs/smoke_repair_20260817/` 为本轮 E2E 验证产物（gitignore 覆盖，保留供复核，可删）。
- v5.0 对 N12 的行数记录（374/352 行）与本轮 AST 实测（26 行）不符，已在 §2.1 订正；后续文档引用 N12 请以本报告为准。
- 昨日报告 `project_repair_report_20260817_k02_f08_f05.md` 与本报告命名不同、并存不冲突。

---

## 9. 文档引用索引

| 引用文档 | 引用章节 | 用途 |
|---|---|---|
| `project_analysis_20260816.md`（v5.0，最新分析文档） | §1 | 严重程度分级沿用 |
| 同上 | §3.2 / §3.3 | F-03 余量设计（STRING/RegNetwork 障碍记录，本轮实测修正） |
| 同上 | §5 | D3 垫片守卫先例（§5 复核补丁的同模式处置依据） |
| 同上 | §6.1 / §6.2 / §6.3 | E2E 验证基线与满足度框架、剩余缺口根因 |
| 同上 | §7.1 / §7.2 | **未解决问题与后续解决策略**（本轮修复指引主依据：TD-BIO-01→F-03 尾巴行、TD-LONG-01→N12 行及排期 4/7/8） |
| 同上 | §8 | 验证命令基线 |
| `project_repair_report_20260817_k02_f08_f05.md`（v1.0） | §3 | F-08 真实解析将在盘状态推进为 parsed（§5 复核补丁根因） |
| 同上 | §5.2 / §5.3 | 昨日遗留债清单与复核差异 |
| `project_repair_report_20260816.md`（v1.0，经 v5.0 转引） | §6.1 / §6.2 | K/F/N 编号体系与分级来源（v5.0 §1 注明的上游出处） |
| v4.0 分析报告（git `f5667db`，经 v5.0 转引） | §6.4 S5/S7 | N06/N10 原始定义 |

---

**报告版本**：v1.0（2026-08-17）
**编制**：主智能体（3 轮修复迭代 + 1 项复核补丁 + E2E 双链路实机复核）
**任务编号映射**：TD-LONG-01 → N12（§2）；TD-BIO-01 → F-03 尾巴（§3）；映射决策依据见 §0
