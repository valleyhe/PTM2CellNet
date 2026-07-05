# PTM2CellNet 代码现状系统性复核报告

**复核日期**：2026-06-30  
**复核范围**：项目根目录 `AGENTS.md`、`docs/` 下规划与报告文档、`src/`、`tests/`、`scripts/`、`configs/`、依赖文件。  
**分析方法**：文档对照、源码结构扫描、静态检查（compileall / pyflakes / flake8 / mypy）、测试覆盖率分析、技术债标记检索。  
**报告原则**：本报告仅客观呈现当前代码现状与存在的问题，不包含任何代码修改建议或解决方案。

> **当前范围提示（2026-07-05）**：本报告是历史快照。实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消，不再作为当前未实现缺口或后续计划项。

---

## 1. 总体结论

1. **核心架构完整性**：`AGENTS.md` 中规定的 21 个核心模块文件（`src/data/loaders.py`、`src/models/architectures.py` 等）均已存在，无结构性缺失。
2. **功能扩展远超原始设计**：代码库在原始架构基础上扩展了 Mamba、DAVF/LatentDAVF、预训练语言模型（ESM-2/ProtBERT/ProtT5）、scVI 桥接、两阶段筛选与 GenKI 集成、变异效应分析、通路分析等大量能力。
3. **文档与代码存在时间差**：部分规划文档（如 DAVF 集成、ESM tokenizer 修复、门控融合、两阶段筛选）仍标记为“待实现”，但对应源码与测试已存在，形成文档滞后于代码的状态。
4. **未实现项较少但明确**：真正未实现的功能主要集中在 `DataLoader.load_from_uniprot()` 接口和 `src/models/roadmap.py` 中声明的 V2 远期功能 stub。
5. **技术债分布广泛**：依赖管理、类型检查、代码风格、测试覆盖、反序列化安全、硬编码路径、异常处理静默吞错等方面存在较多未在现有报告中明确提及的客观问题。

---

## 2. 未实现功能项

### 2.1 AGENTS.md 架构要求中尚未实现的功能

| 需求来源 | 要求接口/功能 | 当前状态 | 说明 |
|---|---|---|---|
| `AGENTS.md` 数据加载器 | `DataLoader.load_from_uniprot(accession_ids: List[str]) -> pd.DataFrame` | **未实现** | `src/data/loaders.py` 实现了 CSV/FASTA/JSON/示例数据加载，但未实现 UniProt 在线/本地批量加载。 |

### 2.2 规划文档中明确标记为待实现的功能（代码层面仍为 stub 或未接入）

| 需求来源 | 功能/接口 | 当前状态 | 说明 |
|---|---|---|---|
| `src/models/roadmap.py` | V2-01 ESM-3 编码器集成 | **未实现** | 仅提供 `raise_documented_error` stub，调用即抛 `NotImplementedError`。 |
| `src/models/roadmap.py` | V2-02 质谱流式（CyTOF）数据支持 | **未实现** | 同上。 |
| `src/models/roadmap.py` | V2-03 自定义 PTM 数据库接入 | **未实现** | 同上。 |
| `src/models/roadmap.py` | V2-04 分布式训练（FSDP/DeepSpeed） | **未实现** | 同上。 |
| `src/models/roadmap.py` | V2-05 图形化用户界面（GUI） | **未实现** | 同上。 |
| `docs/plans/代码修改与测试方案.md` S3 | `MaskedPTMPrediction` 自监督预训练任务 | **未实现** | 文档列为 S3 阶段任务，源码中未见对应实现或测试。 |
| `docs/plans/下一步工作规划.md` | `docs/基准对比报告.md`、`docs/消融实验报告.md`、`docs/可解释性分析报告.md`、`docs/数据统计报告.md` | **未生成** | 规划要求编写的报告文件在仓库中不存在。 |
| `docs/plans/代码修改与测试方案.md` | Sphinx API 文档（`docs/api/`、`docs/conf.py`） | **未实现** | `docs/api/` 目录存在但未见 Sphinx 配置或生成脚本。 |

### 2.3 文档标记为“待实现”但源码已存在的差异项（非未实现，但文档未更新）

以下功能在规划文档中仍被列为“待实现”，但实际源码、测试或脚本已存在，属于文档滞后而非功能缺失。

| 文档标记 | 实际已存在的代码/测试 | 说明 |
|---|---|---|
| ESM2 tokenizer 对齐（`2026-03-15-fix-esm2-tokenizer-mismatch.md`） | `src/data/datasets.py` 中 `ESMTokenizedDataset`、`src/models/pretrained_encoders.py` 中 `ESM2Encoder.tokenize()`、`tests/unit/test_esm_tokenizer.py`、`tests/unit/test_esm_dataset.py`、`tests/unit/test_datamodule_esm.py` | 代码与测试均已实现。 |
| 预训练编码器基类与 ESM2/ProtBERT 编码器（重构设计文档） | `src/models/pretrained_encoders.py` 实现 `PretrainedEncoder`、`ESM2Encoder`、`ProtBERTEncoder`、`ProtT5Encoder` | 已实现。 |
| `GatedPTMFusion`（门控 PTM 融合计划） | `src/models/ptm_modules.py` 中 `GatedPTMFusion`、`tests/unit/test_gated_ptm_fusion.py` | 类与测试已实现；`PTMModule(fusion_type=...)` 的参数化集成状态需结合具体配置验证。 |
| DAVF/LatentDAVF 级联融合（`2026-05-03-davf-integration-plan.md`） | `src/models/davf.py`、`latent_davf.py`、`davf_attention.py`、`biperturb.py`、`delta_predictor.py`、`ptm_direction_mapper.py`、`davf_inference.py`、`davf_checkpoint_utils.py`、`scvi_adapter.py` 以及 `tests/unit/test_davf_inference.py`、`tests/integration/test_davf_pipeline.py` | 核心实现与测试均已存在。 |
| 两阶段筛选与 PTM-aware 软扰动（`2026-03-15-two-stage-screening-...md`） | `src/integration/` 全套模块、`src/evaluation/explainers.py`、`scripts/run_two_stage_explanation.py`、`scripts/run_ptm_virtual_perturbation.py`、`tests/integration/test_two_stage_pipeline.py`、`tests/integration/test_soft_perturbation_pipeline.py` | 实现已存在。 |
| Lightning 训练封装（重构设计文档） | `src/training/lightning_module.py`、`src/data/lightning_datamodule.py`、`src/training/logging_config.py`、`scripts/train_lightning.py`、`tests/unit/training/test_lightning_module.py`、`tests/integration/test_lightning_pipeline.py` | 已实现。 |
| `PTMDirectionMapper`（DAVF 集成设计） | `src/models/ptm_direction_mapper.py`、`tests/unit/test_ptm_direction_mapper.py` | 已实现。 |

---

## 3. 部分实现功能梳理

以下模块或接口并非完全缺失，但存在完成度不足、关键组件缺失或与需求规范不一致的情况。

### 3.1 数据模块（`src/data/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `loaders.py` | 高 | `load_from_uniprot()` 未实现；AGENTS.md 明确要求的 UniProt/PhosphoSitePlus/dbPTM 公共数据库加载未落地。 |
| `schemas.py` | 低 | 仅 9 行基础定义，未形成完整数据契约。 |

### 3.2 模型模块（`src/models/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `roadmap.py` | 占位 | V2-01~V2-05 仅声明为延迟功能 stub。 |
| `predictors.py` — `RegressionPredictor` | 部分 | 前向返回同时包含 `"logits"` 与 `"predictions"`，与 `tests/unit/test_architecture_boundaries.py` 等测试期望的回归输出契约不一致。 |
| `ptm_modules.py` — `PTMModule(fusion_type=...)` | 待验证 | 文档要求支持 `"attention"` 与 `"gated"` 两种融合机制；`GatedPTMFusion` 类已存在，但 `PTMModule` 是否完整暴露 `fusion_type` 参数并与 `architectures.py` 透传需结合配置运行验证。 |
| `scvi_adapter.py` | 高 | 第 366 行在读取 `n_vars` 失败时静默 `pass`，未向上反馈降级状态。 |
| `external_tools.py` | 高 | AlphaFold/BLAST/ClustalW/PSIPRED 客户端已实现并带 fallback，但无测试覆盖。 |
| `biperturb.py`、`davf.py`、`davf_attention.py` | 高 | DAVF 核心实现存在，但无测试文件直接引用，覆盖率空白。 |

### 3.3 API 模块（`src/api/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `app.py` | 高 | 第 62~63 行在 `initialize_variant_workflow()` 失败时捕获所有异常并静默 `pass`，未记录或暴露初始化失败状态。 |
| `routes/` 子包 | 高 | `routes/predictions.py`、`routes/model_info.py`、`routes/state.py` 已实现，但无独立单元测试，仅通过 `tests/unit/test_api.py` 间接引用。 |

### 3.4 分析模块（`src/analysis/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `__init__.py` | 占位 | 仅包含模块 docstring，未导出任何符号，不符合其他子包惯例。 |
| `pathway_integration.py` | 高 | 已实现 KEGG/Reactome/sspa 集成，但 `.coverage` 中无记录。 |
| `extended_pathway_kb.py` | 未知 | 无任何测试引用，未在覆盖率数据中出现。 |

### 3.5 测试与质量

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `tests/.coverage` | 陈旧 | 数据库使用 Windows 绝对路径（`C:\Users\hegu\...`），与当前 Linux 环境不兼容，且仅覆盖 29 个 `src/` 文件，而 `src/` 实际有 87 个 `.py` 文件。 |
| 单元/集成测试 | 不均 | 68 个测试文件覆盖了大量模块，但 `src/api/routes/`、`src/models/davf.py`、`src/models/biperturb.py`、`src/models/davf_attention.py`、`src/models/external_tools.py`、`src/data/extended_pathway_kb.py` 等无任何测试。 |

---

## 4. 技术债识别与分类

以下分类涵盖代码质量、架构缺陷、性能瓶颈、安全隐患、文档缺失、测试覆盖、依赖管理等维度。为便于追溯，已在现有报告（`e2e_status_report_2026_06_30.md`、`CodeReview修复报告.md`、`PTM2CellNet技术深度分析报告.md`）中明确提及的问题标注为“已报告”，其余为“未报告”。

### 4.1 依赖管理

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| `setup.py` 与 `requirements.txt` 依赖不一致 | `setup.py`、`requirements.txt` | `setup.py` 的 `install_requires` 仅列出 11 个包，`requirements.txt` 声明 30+ 个直接依赖，缺失 transformers、lightning、peft、scvi-tools、mamba-ssm 等关键包。 | 未报告 |
| 关键依赖缺失或版本下界矛盾 | `requirements.txt` | `torch>=1.10.0` 与 `lightning>=2.5.0` 并存，后者通常要求 `torch>=2.0`；`torchvision>=0.11.0` 与新版 torch 不兼容。 | 未报告 |
| 无锁定文件 | 项目根目录 | 不存在 `pyproject.toml`、`poetry.lock`、`Pipfile.lock`、`conda-lock.yml` 等锁定文件，依赖可重复性差。 | 未报告 |
| 可选依赖在基础环境未安装 | 当前环境 | `scvi-tools`、`mamba-ssm`、`torch_geometric` 未安装，导致大量测试被跳过或失败。 | 已报告（`torch_geometric` 在 e2e 报告中提及） |

### 4.2 代码质量

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| mypy 类型错误 | 25 个 `src/` 文件，共 59 处 | 错误类型包括 `no-any-return`、`operator`、`arg-type`、`assignment` 等。 | 未报告 |
| 类型提示覆盖率低 | `src/` 全量 | 635 个函数/方法中，78.6% 存在类型提示缺失（返回类型或参数类型）。 | 未报告 |
| 未定义名 `Any` | `src/analysis/pathway_integration.py:229`、`src/models/multitask_ptm.py:381`、`src/models/variant_effect.py:91` | `Any` 被使用但未从 `typing` 导入，存在运行时 `NameError` 风险。 | 未报告 |
| 未使用导入/变量 | 约 25 处 | 例如 `src/models/__init__.py` 导入 `roadmap` 符号但 `pyflakes` 标记为未使用；`src/evaluation/metrics.py` 导入 `math.erfc`/`math.sqrt` 未使用；`src/models/davf.py` 局部变量 `target_u` 未使用。 | 未报告 |
| 行过长 | 9 处 | 超过 120 字符，如 `src/models/architectures.py:367`（145 字符）、`src/integration/genki_adapter.py:99`（140 字符）。 | 未报告 |
| `import` 位置不在顶部 | `src/evaluation/__init__.py:7` | 在创建 `_logger` 后才导入 `.metrics`，触发 E402。 | 未报告 |
| 导入风格混杂 | 多个 `src/` 文件 | 部分模块使用 `from src.xxx import ...` 绝对导入而非相对导入，如 `src/api/routes.py:3`、`src/data/features.py:12`、`src/models/davf.py:19-24`。 | 部分报告（CodeReview 修复了一个脚本，未涉及 `src/` 内部） |
| `print` 替代日志 | 4 个文件 | `src/data/ptm_site_dataset.py`、`src/data/dataset_base.py`、`src/data/validation.py`、`src/models/model_utils.py` 中存在直接 `print` 输出。 | 已报告（CodeReview 列为 Minor Issue） |
| 超长类/函数 | 70 个类、21 个函数超过 80 行 | 例如 `src/models/architectures.py` 中 `PTM2CellNetLarge`（376 行）、`PTM2CellNet`（381 行），`src/models/signaling_network.py` 中 `SignalingNetworkMapper`（476 行）。 | 未报告 |

### 4.3 架构缺陷

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| 抽象基类设计合理但文档未说明 | `src/models/encoders.py`、`src/models/predictors.py` | `SequenceEncoder.forward()` 与 `CellStatePredictor.forward()` 抛 `NotImplementedError`，属于正常抽象接口，非缺陷，但需明确说明。 | 未报告 |
| `src/analysis/__init__.py` 未导出符号 | `src/analysis/__init__.py` | 仅含 docstring，未导出 `pathway_integration`、`gene_mapper` 等公开类。 | 未报告 |
| 设备判断分散 | 7 个以上文件 | `torch.cuda.is_available()` 在 `src/api/app.py`、`src/evaluation/evaluators.py`、`src/training/trainers.py`、`src/models/geneformer_embedding.py` 等模块中重复判断，未统一抽象。 | 未报告 |
| 默认 checkpoint/日志路径重复硬编码 | 多处 | `checkpoints/latent_davf_ibd_norman/best_model.pt`、`outputs/logs` 等默认路径在多个模块中重复出现。 | 未报告 |
| `src/evaluation/__init__.py` 用 `try/except ImportError` 吞掉子模块导入失败 | `src/evaluation/__init__.py` | 失败时仅输出 warning，可能隐藏真实导入错误。 | 未报告 |

### 4.4 性能瓶颈

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| CLI 批量推理非真正 batch | `scripts/predict.py` | 当前批量模式使用 `for row in df.iterrows()` 逐条前向，未将多行堆叠成 batch 统一推理。 | 已报告（e2e 报告） |
| 循环内 `.cpu().numpy()` 转换 | `src/evaluation/evaluators.py`、`src/models/scvi_adapter.py`、`src/integration/genki/perturbation.py`、`src/integration/genki/graph_utils.py` | 批量预测中将张量逐个迁移到 CPU 再转 NumPy，可能形成同步点。 | 未报告 |
| `range(len(...))` 索引循环 | 16 个文件 | 存在基于索引的循环，是否可向量化需结合上下文判断，但属于潜在性能关注点。 | 未报告 |
| 重复 I/O / 默认路径硬编码 | 多处 | 默认 checkpoint 路径在多处重复，易导致重复加载或路径不一致。 | 未报告 |

### 4.5 安全隐患

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| `torch.load(..., weights_only=False)` | `src/api/app.py:50`、`src/models/variant_effect.py:75`、`src/models/davf_checkpoint_utils.py:96`、`src/models/davf_inference.py:214, 228` | 使用旧式加载，PyTorch 未来将默认 `weights_only=True`，存在反序列化安全风险。 | 已报告（e2e 报告） |
| `pickle.load(...)` | `src/analysis/pathway_integration.py:103, 151`、`src/data/validation.py:226` | 加载 pickle 文件，虽带有 `# nosec - cached experiment results from trusted source` 注释，但仍属于反序列化攻击面。 | 未报告 |
| 宽异常捕获并静默吞错 | `src/api/app.py:62`、`src/models/scvi_adapter.py:366`、`src/models/davf_checkpoint_utils.py:145`、`src/models/external_tools.py:164`、`src/integration/genki/reference_data.py:194` | 捕获 `Exception` 后 `pass` 或未记录，可能隐藏运行时错误。 | 部分报告（`src/api/app.py` 与 `scvi_adapter.py` 未在 e2e 中点名，e2e 仅概括为 torch_geometric 缺失） |
| `eval/exec/os.system/subprocess.shell=True` | 未找到 | — | 无 |
| 硬编码密钥/Token | 未找到 | — | 无 |

### 4.6 测试覆盖

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| `.coverage` 数据库陈旧且路径不兼容 | `tests/.coverage` | 使用 Windows 绝对路径，仅覆盖 29 个 `src/` 文件，而实际有 87 个文件。 | 未报告 |
| 8 个 `src/` 文件完全无测试 | 见下表 | 无任何测试 import 或覆盖率记录。 | 未报告 |
| 可选依赖导致大量测试跳过 | 多个测试文件 | ESM/ProtBERT/CUDA/ONNX/scvi-tools/peft/torch_geometric 等环境缺失时，大量 `skipif` 测试被跳过。 | 部分报告（e2e 报告提及 torch_geometric 相关失败） |

**完全无测试覆盖的 `src/` 文件**：

| 文件 | 说明 |
|---|---|
| `src/api/routes/model_info.py` | API 健康检查 / 模型信息路由 |
| `src/api/routes/predictions.py` | API 预测、批量预测、变异预测路由 |
| `src/api/routes/state.py` | API 状态初始化路由 |
| `src/data/extended_pathway_kb.py` | 扩展通路知识库 |
| `src/models/biperturb.py` | BiPerturb 编码器 |
| `src/models/davf.py` | DAVF 核心模型 |
| `src/models/davf_attention.py` | DAVF 方向感知注意力 |
| `src/models/external_tools.py` | AlphaFold/BLAST/ClustalW/PSIPRED 客户端 |

### 4.7 文档缺失

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| Sphinx API 文档未建立 | `docs/api/`、`docs/conf.py` | `docs/api/` 目录存在，但无 `conf.py` 或生成脚本。 | 未报告 |
| 规划报告未生成 | `docs/` | `docs/基准对比报告.md`、`docs/消融实验报告.md`、`docs/可解释性分析报告.md`、`docs/数据统计报告.md` 不存在。 | 未报告 |
| `CHANGELOG.md` 状态未核实 | 项目根目录 | 规划要求维护，但未在本次复核中确认其存在与完整性。 | 未报告 |
| 文档滞后于代码 | `docs/superpowers/`、`docs/plans/` | 多个文档仍将已实现的 ESM2 tokenizer、GatedPTMFusion、DAVF 集成、两阶段筛选标记为“待实现”。 | 未报告 |

---

## 5. 现有报告中已明确提及的问题汇总

为避免重复归类，以下问题已在现有报告中提及，本次复核仅做索引：

| 问题 | 已提及报告 |
|---|---|
| `RegressionPredictor` 回归输出同时包含 `"logits"` 与 `"predictions"` | `docs/e2e_status_report_2026_06_30.md` |
| `torch_geometric` 缺失导致 GenKI / 两阶段流程失败 | `docs/e2e_status_report_2026_06_30.md` |
| DAVF 旧版 checkpoint 警告文案不含 `"UNSAFE"` | `docs/e2e_status_report_2026_06_30.md` |
| `scripts/predict.py` 批量模式未做真正 batch 推理 | `docs/e2e_status_report_2026_06_30.md` |
| `torch.load(weights_only=False)` 告警 | `docs/e2e_status_report_2026_06_30.md` |
| 训练曲线 legend 单 epoch 警告 | `docs/e2e_status_report_2026_06_30.md` |
| `scripts/data_statistics.py` 异常处理过于宽泛 | `docs/CodeReview修复报告.md`（已修复） |
| `scripts/test_mamba_standalone.py` 不规范导入 | `docs/CodeReview修复报告.md`（已修复） |
| Token 编码不一致 | `docs/CodeReview修复报告.md`（已修复） |
| 模型尺寸解析边界 | `docs/CodeReview修复报告.md`（已修复） |
| 建议使用 `logging` 替代 `print` | `docs/CodeReview修复报告.md`（列为 Minor） |
| 建议统一 docstring 格式与增加类型注解 | `docs/CodeReview修复报告.md`（列为 Minor） |

---

## 6. 附录：数据源与工具输出

- **源码扫描**：`src/` 下 87 个 Python 文件，总计约 22,138 行代码。
- **静态检查**：
  - `python -m compileall -q src tests scripts`：通过，无语法错误。
  - `pyflakes src scripts`：发现 30+ 处未使用导入/变量/未定义名。
  - `flake8 src --max-line-length=120`：30+ 处问题（E501/F401/F821/F841/E402）。
  - `mypy src/ --ignore-missing-imports`：59 处类型错误，分布于 25 个文件。
  - AST 扫描：635 个函数/方法，78.6% 存在类型提示缺失；70 个类、21 个函数超过 80 行。
- **测试扫描**：`tests/` 下 71 个 Python 文件，其中 68 个为测试文件；`tests/.coverage` 覆盖 29 个 `src/` 文件。
- **技术债标记**：Grep 检索 `TODO`、`FIXME`、`XXX`、`HACK`、`NotImplementedError`、`DEPRECATED`、`pragma: no cover`、`pass` 等。

---

**报告生成时间**：2026-06-30  
**报告文件**：`docs/code_status_systematic_review_report.md`
