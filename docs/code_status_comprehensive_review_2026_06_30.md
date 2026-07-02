# PTM2CellNet 代码现状系统性复核报告

**复核日期**：2026-06-30  
**复核范围**：项目根目录、`src/`、`tests/`、`scripts/`、`configs/`、`docs/`、依赖文件及架构设计文档 `AGENTS.md`。  
**分析方法**：文档对照、源码结构扫描、依赖与配置审查、现有内部报告交叉验证、专项探索代理复核。  
**报告原则**：本报告仅客观呈现当前代码现状与存在的问题，不包含任何代码修改建议或解决方案。

---

## 1. 总体结论

1. **核心架构完整性**：`AGENTS.md` 中规定的 21 个核心模块文件（`src/data/loaders.py`、`src/models/architectures.py` 等）均已存在，无结构性缺失。
2. **功能扩展远超原始设计**：代码库在原始架构基础上扩展了 Mamba、DAVF/LatentDAVF、预训练语言模型（ESM-2/ProtBERT/ProtT5）、scVI 桥接、两阶段筛选与 GenKI 集成、变异效应分析、通路分析等大量能力。
3. **文档与代码存在时间差**：部分规划文档（如 DAVF 集成、ESM tokenizer 修复、门控融合、两阶段筛选）仍标记为“待实现”，但对应源码与测试已存在，形成文档滞后于代码的状态。
4. **未实现项较少但明确**：真正未实现的功能主要集中在公共 PTM 数据库加载（PhosphoSitePlus / dbPTM / CPLM）、`src/models/roadmap.py` 中声明的 V2 远期功能 stub、`MaskedPTMPrediction` 自监督预训练任务、HDF5 IO、结构特征预测、数据增强接入主 Dataset、Sphinx API 文档及部分规划报告交付物。
5. **技术债分布广泛**：依赖管理、类型检查、代码风格、测试覆盖、反序列化安全、硬编码路径、异常处理静默吞错等方面存在较多未在现有报告中明确提及的客观问题。

---

## 2. 未实现功能项

### 2.1 AGENTS.md / README 架构要求中尚未实现的功能

| 需求来源 | 要求接口/功能 | 当前状态 | 说明 |
|---|---|---|---|
| `AGENTS.md` 数据加载器 — PhosphoSitePlus | `DataLoader.load_from_phosphositeplus(...)` 或等价批量下载/解析接口 | **未实现** | `src/data/loaders.py` 实现了 CSV/FASTA/JSON/UniProt 加载，但未实现 PhosphoSitePlus 数据加载。 |
| `AGENTS.md` 数据加载器 — dbPTM | `DataLoader.load_from_dbptm(...)` 或等价批量下载/解析接口 | **未实现** | 同上。 |
| `AGENTS.md` 数据加载器 — CPLM | `DataLoader.load_from_cplm(...)` 或等价赖氨酸修饰数据加载接口 | **未实现** | 同上。 |
| `AGENTS.md` 目录结构 | `configs/model/` 模型配置目录 | **未实现** | 目录存在但为空。 |
| `AGENTS.md` API 路由 | `src/api/routes.py` 直接定义 `/predict`、`/batch_predict`、`/health`、`/model_info` 等端点 | **部分实现** | 端点已迁移至 `src/api/routes/` 子包，`routes.py` 仅做符号转发。 |

> 注：`DataLoader.load_from_uniprot()` 已在 `src/data/loaders.py:149` 实现，并带有批量请求与回退逻辑。

### 2.2 规划文档中明确标记为待实现的功能（代码层面仍为 stub 或未接入）

| 需求来源 | 功能/接口 | 当前状态 | 说明 |
|---|---|---|---|
| `src/models/roadmap.py` | V2-01 ESM-3 编码器集成 | **未实现** | 仅提供 `raise_documented_error` stub，调用即抛 `NotImplementedError`。 |
| `src/models/roadmap.py` | V2-02 质谱流式（CyTOF）数据支持 | **未实现** | 同上。 |
| `src/models/roadmap.py` | V2-03 自定义 PTM 数据库接入 | **未实现** | 同上。 |
| `src/models/roadmap.py` | V2-04 分布式训练（FSDP/DeepSpeed） | **未实现** | 同上。 |
| `src/models/roadmap.py` | V2-05 图形化用户界面（GUI） | **未实现** | 同上。 |
| `docs/plans/代码修改与测试方案.md` S3 | `MaskedPTMPrediction` 自监督预训练任务 | **未实现** | 文档列为 S3 阶段任务，`src/training/self_supervised.py` 不存在。 |
| `docs/Mamba验证报告.md` 附录 A | `scripts/test_mamba_standalone.py` | **未实现** | Mamba 独立测试脚本不存在。 |
| `docs/plans/下一步工作规划.md` | `docs/数据统计报告.md`、`docs/消融实验报告.md`、`docs/可解释性分析报告.md` | **未生成** | 规划要求编写的报告文件在仓库中不存在。 |
| `docs/plans/下一步工作规划.md` | `docs/基准对比报告.md`（中文） | **未生成** | 英文版 `docs/benchmark_report.md` 已存在，中文版未生成。 |
| `docs/plans/代码修改与测试方案.md` | Sphinx API 文档（`docs/conf.py`、`docs/api/` 下按模块组织的 `.rst`/`.md`） | **未实现** | `docs/api/` 目录存在但为空，无 `docs/conf.py` 或生成脚本。 |

### 2.3 文档标记为“待实现”但源码已存在的差异项（非未实现，但文档未更新）

以下功能在规划文档中仍被列为“待实现”，但实际源码、测试或脚本已存在，属于文档滞后而非功能缺失。

| 文档标记 | 实际已存在的代码/测试 | 说明 |
|---|---|---|
| ESM2 tokenizer 对齐 | `src/data/datasets.py` 中 `ESMTokenizedDataset`、`src/models/pretrained_encoders.py` 中 `ESM2Encoder.tokenize()`、`tests/unit/test_esm_tokenizer.py`、`tests/unit/test_esm_dataset.py`、`tests/unit/test_datamodule_esm.py` | 代码与测试均已实现。 |
| 预训练编码器基类与 ESM2/ProtBERT 编码器 | `src/models/pretrained_encoders.py` 实现 `PretrainedEncoder`、`ESM2Encoder`、`ProtBERTEncoder`、`ProtT5Encoder` | 已实现。 |
| `GatedPTMFusion` 门控 PTM 融合 | `src/models/ptm_modules.py` 中 `GatedPTMFusion`、`tests/unit/test_gated_ptm_fusion.py` | 类与测试已实现。 |
| DAVF/LatentDAVF 级联融合 | `src/models/davf.py`、`latent_davf.py`、`davf_attention.py`、`biperturb.py`、`delta_predictor.py`、`ptm_direction_mapper.py`、`davf_inference.py`、`davf_checkpoint_utils.py`、`scvi_adapter.py` 以及对应测试 | 核心实现与测试均已存在。 |
| 两阶段筛选与 PTM-aware 软扰动 | `src/integration/` 全套模块、`src/evaluation/explainers.py`、`scripts/run_two_stage_explanation.py`、`scripts/run_ptm_virtual_perturbation.py`、对应集成测试 | 实现已存在。 |
| Lightning 训练封装 | `src/training/lightning_module.py`、`src/data/lightning_datamodule.py`、`src/training/logging_config.py`、`scripts/train_lightning.py`、对应测试 | 已实现。 |
| `PTMDirectionMapper` | `src/models/ptm_direction_mapper.py`、`tests/unit/test_ptm_direction_mapper.py` | 已实现。 |

### 2.4 配置文件中声明但代码未使用/未充分使用的配置项

| 配置项 | 配置文件 | 声明用途 | 当前代码状态 | 涉及文件 |
|---|---|---|---|---|
| `features.sequence_encoding` | `configs/default.yaml` | 控制序列特征编码方式（onehot/kmer/both） | 未被主训练/推理数据流使用；仅在 `FeatureExtractor` 内部读取 | `configs/default.yaml:38` |
| `features.kmer_size` | `configs/default.yaml` | k-mer 大小 | 同上 | `configs/default.yaml:39` |
| `features.include_physicochemical` | `configs/default.yaml` | 是否包含理化特征 | 同上 | `configs/default.yaml:40` |
| `features.include_ptm_features` | `configs/default.yaml` | 是否包含 PTM 特征 | 同上 | `configs/default.yaml:41` |
| `monitoring.metrics_enabled` | `configs/production.yaml` | 是否启用监控指标 | 代码中无引用 | `configs/production.yaml:38` |
| `monitoring.metrics_port` | `configs/production.yaml` | 监控指标端口 | 代码中无引用 | `configs/production.yaml:39` |
| `monitoring.health_check_interval` | `configs/production.yaml` | 健康检查间隔 | 代码中无引用 | `configs/production.yaml:40` |

---

## 3. 部分实现功能梳理

以下模块或接口并非完全缺失，但存在完成度不足、关键组件缺失或与需求规范不一致的情况。

### 3.1 数据模块（`src/data/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `loaders.py` | 高 | UniProt 加载已实现；PhosphoSitePlus / dbPTM / CPLM 公共 PTM 数据库加载未实现。 |
| `schemas.py` | 中 | 当前 58 行，定义了 6 个 dataclass 及 `__all__`，但未形成完整端到端数据契约。 |
| `features.py` | 中 | `FeatureExtractor` 已实现 one-hot/k-mer/理化/PTM 特征提取，但 `configs/default.yaml` 中 `features.*` 配置未被主训练脚本与 `PTMDataset.__getitem__` 消费。 |

### 3.2 模型模块（`src/models/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `roadmap.py` | 占位 | V2-01~V2-05 仅声明为延迟功能 stub。 |
| `predictors.py` — `RegressionPredictor` | 中 | 当前 `forward()` 仅返回 `{"predictions": predictions}`（`src/models/predictors.py:72-75`），需确认是否与所有调用方期望的回归输出契约一致。 |
| `ptm_modules.py` — `PTMModule(fusion_type=...)` | 待验证 | 文档要求支持 `"attention"` 与 `"gated"` 两种融合机制；`GatedPTMFusion` 类已存在，但 `PTMModule` 是否完整暴露 `fusion_type` 参数并与 `architectures.py` 透传需结合配置运行验证。 |
| `scvi_adapter.py` | 高 | 在读取 `n_vars` 失败时静默 `pass`，未向上反馈降级状态。 |
| `external_tools.py` | 高 | AlphaFold/BLAST/ClustalW/PSIPRED 客户端已实现并带 fallback，已有对应测试覆盖。 |
| `biperturb.py`、`davf.py`、`davf_attention.py` | 高 | DAVF 核心实现存在，但无测试文件直接引用，覆盖率空白。 |

### 3.3 API 模块（`src/api/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `app.py` | 高 | 在 `initialize_variant_workflow()` 失败时捕获所有异常并静默 `pass`，未记录或暴露初始化失败状态。 |
| `routes/` 子包 | 高 | `routes/predictions.py`、`routes/model_info.py`、`routes/state.py` 已实现，并已有对应测试覆盖。 |

### 3.4 分析模块（`src/analysis/`）

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `__init__.py` | 高 | 当前已导出 `PathwayDatabaseIntegration`、`GeneMapper`、`HGVSVariantParser`、`VariantEffectWorkflow` 等符号（`src/analysis/__init__.py:70-81`），但采用 `try/except ImportError` 封装，导入失败时仅输出 warning 并置为 `None`。 |
| `pathway_integration.py` | 高 | 已实现 KEGG/Reactome/sspa 集成，但覆盖率数据未记录。 |
| `extended_pathway_kb.py` | 低 | 文件为大型静态数据字典集合，缺少查询/检测类接口，且无测试覆盖。 |

### 3.5 测试与质量

| 文件/功能 | 完成度 | 缺失或不符合规范的部分 |
|---|---|---|
| `tests/.coverage` | 陈旧 | 数据库使用 Windows 绝对路径，与当前 Linux 环境不兼容，且仅覆盖部分 `src/` 文件。 |
| 单元/集成测试 | 不均 | 68 个测试文件覆盖了大量模块，但 `src/models/biperturb.py`、`src/models/davf.py`、`src/models/davf_attention.py`、`src/data/extended_pathway_kb.py` 等无任何测试。 |

### 3.6 与 AGENTS.md 接口契约的主要偏差

| 契约项 | 现状 | 偏差 |
|---|---|---|
| `DataLoader` 外部数据库加载 | 仅 UniProt 已实现 | PhosphoSitePlus、dbPTM、CPLM 未实现 |
| `FeatureExtractor.extract_ptm_features` | 输入 JSON 字符串，返回字典 | 与 AGENTS.md 的 `List[Dict] -> np.ndarray` 不一致 |
| `PTMDataModule` 为 Lightning DataModule | `src/data/datasets.py` 中的 `PTMDataModule` 为非 Lightning 版本；Lightning 版本为 `src/data/lightning_datamodule.py` 中的 `PTMLightningDataModule` | 命名冲突，契约不唯一 |
| `Trainer.validate` | 未实现 | 缺失；当前验证通过 `_val_epoch` 在 `fit` 内部完成 |
| `ModelCheckpoint` / `EarlyStopping` | 自定义 `Callback` 子类 | 不是 PyTorch Lightning 的 `pl.Callback` |
| IO 支持 HDF5 | 未实现 | 缺失 |
| 数据增强接入主 Dataset | `src/data/augmentation.py` 存在但 `PTMDataset` 未调用 | 缺失 |
| 结构特征预测 | 未实现 | 缺失 |

---

## 4. 技术债识别与分类

以下分类涵盖代码质量、架构缺陷、性能瓶颈、安全隐患、文档缺失、测试覆盖、依赖管理等维度。为便于追溯，已在现有报告（`e2e_status_report_2026_06_30.md`、`CodeReview修复报告.md`、`PTM2CellNet技术深度分析报告.md`）中明确提及的问题标注为“已报告”，其余为“未报告”。

### 4.1 依赖管理

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| `setup.py` 与 `requirements.txt` 依赖不一致 | `setup.py`、`requirements.txt` | `setup.py` 的 `install_requires` 仅列出 11 个包，`requirements.txt` 声明 30+ 个直接依赖，缺失 transformers、lightning、peft、scvi-tools、mamba-ssm 等关键包。 | 未报告 |
| 关键依赖版本下界矛盾 | `requirements.txt` | `torch>=1.10.0` 与 `lightning>=2.5.0` 并存，后者通常要求 `torch>=2.0`；`torchvision>=0.11.0` 与新版 torch 不兼容。 | 未报告 |
| 无锁定文件 | 项目根目录 | 不存在 `pyproject.toml`、`poetry.lock`、`Pipfile.lock`、`conda-lock.yml` 等锁定文件，依赖可重复性差。 | 未报告 |
| 可选依赖在基础环境未安装 | 当前环境 | `scvi-tools`、`mamba-ssm`、`torch_geometric` 未安装，导致大量测试被跳过或失败。 | 已报告 |
| `torch_geometric` 未声明 | `requirements.txt`、`setup.py` | GenKI 后端需要 `torch_geometric`，但依赖文件中未列出。 | 未报告 |
| `fair-esm` 未声明 | `scripts/ensemble_predict.py:90`、`scripts/batch_predict.py:134` | 直接 `import esm`，但未在 `requirements.txt` 或 `setup.py` 中声明。 | 未报告 |
| 脚本导入方式不透明 | 多个 `scripts/*.py` | 各脚本自行 `sys.path.insert` 并在函数内延迟导入，未通过安装后的包名导入。 | 未报告 |

### 4.2 代码质量

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| mypy 类型错误 | 25 个 `src/` 文件，共 59 处 | 错误类型包括 `no-any-return`、`operator`、`arg-type`、`assignment` 等。 | 未报告 |
| 类型提示覆盖率低 | `src/` 全量 | 大量函数/方法存在返回类型或参数类型缺失。 | 未报告 |
| 未定义名 `Any` | `src/analysis/pathway_integration.py:229`、`src/models/multitask_ptm.py:381`、`src/models/variant_effect.py:91` | `Any` 被使用但未从 `typing` 导入，存在运行时 `NameError` 风险。 | 未报告 |
| 未使用导入/变量 | 约 25 处 | 例如 `src/models/__init__.py` 导入 `roadmap` 符号但未使用；`src/evaluation/metrics.py` 导入 `math.erfc`/`math.sqrt` 未使用；`src/models/davf.py` 局部变量未使用。 | 未报告 |
| 行过长 | 9 处 | 超过 120 字符，如 `src/models/architectures.py:367`（145 字符）、`src/integration/genki_adapter.py:99`（140 字符）。 | 未报告 |
| `import` 位置不在顶部 | `src/evaluation/__init__.py:7` | 在创建 `_logger` 后才导入 `.metrics`，触发 E402。 | 未报告 |
| 导入风格混杂 | 多个 `src/` 文件 | 部分模块使用 `from src.xxx import ...` 绝对导入而非相对导入，如 `src/api/routes.py:3`、`src/data/features.py:12`、`src/models/davf.py:19-24`。 | 部分报告 |
| `print` 替代日志 | 4 个文件 | `src/data/ptm_site_dataset.py`、`src/data/dataset_base.py`、`src/data/validation.py`、`src/models/model_utils.py` 中存在直接 `print` 输出。 | 已报告 |
| 超长类/函数 | 70 个类、21 个函数超过 80 行 | 例如 `src/models/architectures.py` 中 `PTM2CellNetLarge`（376 行）、`PTM2CellNet`（381 行），`src/models/signaling_network.py` 中 `SignalingNetworkMapper`（476 行）。 | 未报告 |
| 脚本硬编码默认路径 | 多个 `scripts/*.py` | `scripts/predict.py`、`scripts/evaluate.py`、`scripts/train.py` 等默认参数硬编码 `outputs/models/best_model.pt`、`configs/default.yaml`、`outputs/results` 等路径。 | 未报告 |
| 魔法数字 | `src/api/routes/predictions.py:194/204/349/356` | 通路活性阈值 `0.5`、`delta_prob` 默认 `1.0` 等未参数化。 | 未报告 |
| 脚本静默吞错 | `scripts/integrate_data_v2.py:206-207`、`scripts/finetune_davf.py:292-293` | `except Exception: pass` 隐藏网络/解析/指标计算错误。 | 未报告 |
| 函数嵌套定义 | `scripts/train.py:169-176` | `_sanitize_nans` 作为 `main()` 内部嵌套函数，降低可测试性。 | 未报告 |
| 脚本类型注解缺失 | 多个 `scripts/*.py` | `parse_args()`、`main()` 及辅助函数普遍缺少类型注解。 | 未报告 |

### 4.3 架构缺陷

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| 抽象基类文档未说明 | `src/models/encoders.py`、`src/models/predictors.py` | `SequenceEncoder.forward()` 与 `CellStatePredictor.forward()` 抛 `NotImplementedError`，属于正常抽象接口，但缺少文档说明。 | 未报告 |
| 设备判断分散 | 7 个以上文件 | `torch.cuda.is_available()` 在 `src/api/app.py`、`src/evaluation/evaluators.py`、`src/training/trainers.py`、`src/models/geneformer_embedding.py` 等模块中重复判断，未统一抽象。 | 未报告 |
| 默认 checkpoint/日志路径重复硬编码 | 多处 | `checkpoints/latent_davf_ibd_norman/best_model.pt`、`outputs/logs` 等默认路径在多个模块中重复出现。 | 未报告 |
| `src/evaluation/__init__.py` 用 `try/except ImportError` 吞掉子模块导入失败 | `src/evaluation/__init__.py` | 失败时仅输出 warning，可能隐藏真实导入错误。 | 未报告 |
| `src/analysis/__init__.py` 用 `try/except ImportError` 吞掉子模块导入失败 | `src/analysis/__init__.py` | 失败时仅输出 warning 并将符号置为 `None`，可能隐藏真实导入错误。 | 未报告 |
| 配置与实现脱节 | `configs/default.yaml`、`configs/production.yaml` | `features.*` 与 `monitoring.*` 配置在代码中未被主流程消费。 | 未报告 |
| `PTMDataModule` 命名冲突 | `src/data/datasets.py`、`src/data/lightning_datamodule.py` | 存在非 Lightning 与 Lightning 两个同名概念，AGENTS.md 契约不唯一。 | 未报告 |
| 脚本导入方式不统一 | 多个 `scripts/*.py` | 各脚本自行 `sys.path.insert` 或 `_ensure_project_root()`，未通过安装后的包名导入。 | 未报告 |
| 单一函数职责过重 | `src/api/routes/predictions.py:32-94` | `preprocess_request()` 同时承担序列验证、PTM 解析、张量构造与全局状态访问。 | 未报告 |

### 4.4 性能瓶颈

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| CLI 批量推理非真正 batch | `scripts/predict.py` | 当前批量模式使用 `for row in df.iterrows()` 逐条前向，未将多行堆叠成 batch 统一推理；API `/batch_predict` 已实现真正 batch。 | 已报告 |
| 循环内 `.cpu().numpy()` 转换 | `src/evaluation/evaluators.py`、`src/models/scvi_adapter.py`、`src/integration/genki/perturbation.py`、`src/integration/genki/graph_utils.py` | 批量预测中将张量逐个迁移到 CPU 再转 NumPy，可能形成同步点。 | 未报告 |
| `range(len(...))` 索引循环 | 16 个文件 | 存在基于索引的循环，是否可向量化需结合上下文判断，但属于潜在性能关注点。 | 未报告 |
| 重复 I/O / 默认路径硬编码 | 多处 | 默认 checkpoint 路径在多处重复，易导致重复加载或路径不一致。 | 未报告 |
| `scripts/batch_predict.py` 未真正 batch 化 | `scripts/batch_predict.py` | 类名虽为 `BatchPTMPredictor`，但内部对每条序列独立 tokenize/前向。 | 未报告 |
| 循环内 `.cpu().numpy()` 设备同步 | `scripts/ensemble_predict.py:141-147` | 对每模型、每 batch 在循环中调用 `.cpu().numpy()`，形成 CPU/GPU 同步点。 | 未报告 |
| 低效 Python 级循环 | `src/data/features.py:98`、`src/data/homology_splitter.py:37-38` 等 | 使用 `range(len(...))` 索引循环，未向量化。 | 未报告 |
| 模型句柄未缓存 | `scripts/validate_cptac.py:231-250` | 对每个 PTM 类型分别扫描目录并加载 checkpoint，未缓存模型句柄。 | 未报告 |

### 4.5 安全隐患

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| `torch.load(..., weights_only=False)` | `src/api/app.py:50`、`src/models/variant_effect.py:75`、`src/models/davf_checkpoint_utils.py:96`、`src/models/davf_inference.py:214, 228` | 使用旧式加载，PyTorch 未来将默认 `weights_only=True`，存在反序列化安全风险。 | 已报告 |
| `pickle.load(...)` | `src/analysis/pathway_integration.py:103, 151`、`src/data/validation.py:226` | 加载 pickle 文件，虽带有 `# nosec` 注释，但仍属于反序列化攻击面。 | 未报告 |
| 宽异常捕获并静默吞错 | `src/api/app.py:62`、`src/models/scvi_adapter.py:366`、`src/models/davf_checkpoint_utils.py:145`、`src/models/external_tools.py:164`、`src/integration/genki/reference_data.py:194` | 捕获 `Exception` 后 `pass` 或未记录，可能隐藏运行时错误。 | 部分报告 |
| `torch.load` 未指定 `weights_only` | `scripts/batch_predict.py:122/147`、`scripts/validate_cptac.py:250`、`scripts/train_multitask_ptm.py:291`、`scripts/ensemble_predict.py:116`、`tests/unit/test_training_inference_consistency.py:82/144/584` | 使用默认 pickle 反序列化，存在安全风险。 | 未报告 |
| CLI 参数校验缺失 | `scripts/ensemble_predict.py:289-355`、`scripts/predict_variant_effect.py:56-57` | `--models`/`--types`/`--weights` 长度未校验；`--variant` 格式与 README 示例不一致且无校验。 | 未报告 |
| `eval/exec/os.system/subprocess.shell=True` | 未找到 | — | 无 |
| 硬编码密钥/Token | 未找到 | — | 无 |

### 4.6 测试覆盖

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| `.coverage` 数据库陈旧且路径不兼容 | `tests/.coverage` | 使用 Windows 绝对路径，覆盖范围有限。 | 未报告 |
| 4 个 `src/` 文件完全无测试 | 见下表 | 无任何测试 import 或覆盖率记录。 | 未报告 |
| 可选依赖导致大量测试跳过 | 多个测试文件 | ESM/ProtBERT/CUDA/ONNX/scvi-tools/peft/torch_geometric 等环境缺失时，大量 `skipif` 测试被跳过。 | 部分报告 |
| 脚本测试覆盖不足 | `scripts/ensemble_predict.py`、`scripts/batch_predict.py`、`scripts/validate_cptac.py`、`scripts/train_multitask_ptm.py`、`scripts/integrate_data_v2.py`、`scripts/finetune_davf.py` | 大量脚本无对应单元/集成测试，关键 CLI 路径未覆盖。 | 未报告 |
| 占位测试未实现 | `tests/unit/test_api_routes.py:309` | `TestWithRealWeights.test_predict_with_production_model` 仅包含 `pass` 占位。 | 未报告 |
| 条件分支未覆盖 | `src/api/routes/predictions.py` | `/predict/variant` 与 `/batch_predict` 的错误分支、通路分析失败分支缺乏专门测试。 | 未报告 |

**完全无测试覆盖的 `src/` 文件**：

| 文件 | 说明 |
|---|---|
| `src/data/extended_pathway_kb.py` | 扩展通路知识库 |
| `src/models/biperturb.py` | BiPerturb 编码器 |
| `src/models/davf.py` | DAVF 核心模型 |
| `src/models/davf_attention.py` | DAVF 方向感知注意力 |

### 4.7 文档缺失

| 问题 | 位置 | 说明 | 报告状态 |
|---|---|---|---|
| Sphinx API 文档未建立 | `docs/api/`、`docs/conf.py` | `docs/api/` 目录存在但为空，无 `docs/conf.py` 或生成脚本。 | 未报告 |
| 规划报告未生成 | `docs/` | `docs/基准对比报告.md`（中文）、`docs/消融实验报告.md`、`docs/可解释性分析报告.md`、`docs/数据统计报告.md` 不存在。 | 未报告 |
| 文档滞后于代码 | `docs/superpowers/`、`docs/plans/` | 多个文档仍将已实现的 ESM2 tokenizer、GatedPTMFusion、DAVF 集成、两阶段筛选标记为“待实现”。 | 未报告 |
| README 与代码不符 | `README.md:98` | 示例 `--variant "BRAF_V600E"` 与脚本实际要求的 `UniProtID:Position:RefAA:AltAA` 格式不一致。 | 未报告 |
| 大量公共接口缺少 docstring | `src/models/predictors.py`、`src/models/encoders.py`、`src/models/ptm_modules.py`、`src/data/validation.py`、`src/data/dataset_base.py`、`src/evaluation/explainers.py`、`src/integration/genki_adapter.py`、`src/training/peft_config.py` 等 | 核心类与公共函数缺少文档字符串。 | 未报告 |

> 注：`CHANGELOG.md`、`Dockerfile`、`docker-compose.yml`、`configs/production.yaml`、`configs/training/finetune_binary.yaml` 已存在。

---

## 5. 现有报告中已明确提及的问题索引

为避免重复归类，以下问题已在现有报告中提及，本次复核仅做索引：

| 问题 | 已提及报告 |
|---|---|
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

## 6. 附录：数据源与统计

- **源码扫描**：`src/` 下 87 个 Python 文件，总计约 22,361 行代码。
- **脚本扫描**：`scripts/` 下 22 个 Python 文件。
- **测试扫描**：`tests/` 下 76 个 Python 文件，总计约 17,060 行代码；其中 68 个为测试文件。
- **技术债标记**：Grep 检索 `TODO`、`FIXME`、`XXX`、`HACK`、`NotImplementedError`、`pragma: no cover`、`pass` 等，得到 `TODO/FIXME/XXX/HACK` 标记 2 处、`NotImplementedError`/`raise_documented_error` 6 处、`pragma: no cover` 5 处。
- **参考报告**：本次复核交叉验证了 `docs/code_status_systematic_review_report.md`、`docs/e2e_status_report_2026_06_30.md`、`docs/CodeReview修复报告.md`、`docs/PTM2CellNet技术深度分析报告.md` 等内部文档。

---

**报告生成时间**：2026-06-30  
**报告文件**：`docs/code_status_comprehensive_review_2026_06_30.md`

---

## 附录 A：未实现功能项详细复核（探索代理输出）

# PTM2CellNet 未实现功能项复核报告

**复核范围**：`AGENTS.md`、`README.md`、`docs/` 关键文档、`src/`、`scripts/`、`tests/`、`configs/`  
**复核日期**：2026-06-30  
**说明**：本报告仅客观记录现状，不含修改建议。

---

## 1. 完全未实现的功能项

### 1.1 AGENTS.md / README 架构要求中的外部数据库加载

| 功能项 | 来源文档 | 预期实现 | 当前状态 | 涉及文件 |
|--------|----------|----------|----------|----------|
| PhosphoSitePlus 数据加载 | `AGENTS.md` 外部工具/API 集成表 | `DataLoader.load_from_phosphositeplus(...)` 或等价批量下载/解析接口 | 未实现 | `src/data/loaders.py` |
| dbPTM 数据加载 | `AGENTS.md` 外部工具/API 集成表 | `DataLoader.load_from_dbptm(...)` 或等价批量下载/解析接口 | 未实现 | `src/data/loaders.py` |
| CPLM 赖氨酸修饰数据加载 | `AGENTS.md` 外部工具/API 集成表 | `DataLoader.load_from_cplm(...)` 或等价批量下载/解析接口 | 未实现 | `src/data/loaders.py` |
| `configs/model/` 模型配置目录 | `AGENTS.md` 目录结构 | 存放各模型架构专属 YAML 配置 | 目录为空 | `configs/model/` |

> 注：`DataLoader.load_from_uniprot()` 已在 `src/data/loaders.py:149` 实现，并带有批量请求与回退逻辑。

### 1.2 规划文档要求的自监督预训练任务

| 功能项 | 来源文档 | 预期实现 | 当前状态 | 涉及文件 |
|--------|----------|----------|----------|----------|
| `MaskedPTMPrediction` 自监督预训练 | `docs/plans/代码修改与测试方案.md` S3 阶段 | 随机 mask PTM 位点并让模型预测被 mask 的 PTM 类型，对应 `src/training/self_supervised.py` | 未实现 | `src/training/self_supervised.py` 不存在 |

### 1.3 规划文档要求的报告/脚本交付物

| 功能项 | 来源文档 | 预期实现 | 当前状态 | 涉及文件 |
|--------|----------|----------|----------|----------|
| 数据统计报告 | `docs/下一步工作规划.md` 1.1 | `docs/数据统计报告.md` | 未生成 | `docs/数据统计报告.md` 不存在 |
| 消融实验报告 | `docs/下一步工作规划.md` 2.2 | `docs/消融实验报告.md` | 未生成 | `docs/消融实验报告.md` 不存在 |
| 可解释性分析报告 | `docs/下一步工作规划.md` 3.3 | `docs/可解释性分析报告.md` | 未生成 | `docs/可解释性分析报告.md` 不存在 |
| Mamba 独立测试脚本 | `docs/Mamba验证报告.md` 附录 A | `scripts/test_mamba_standalone.py` | 未实现 | `scripts/test_mamba_standalone.py` 不存在 |
| 基准对比报告（中文） | `docs/下一步工作规划.md` 1.3 | `docs/基准对比报告.md` | 未生成 | `docs/基准对比报告.md` 不存在；英文版 `docs/benchmark_report.md` 已存在 |

### 1.4 Sphinx API 文档

| 功能项 | 来源文档 | 预期实现 | 当前状态 | 涉及文件 |
|--------|----------|----------|----------|----------|
| Sphinx 配置 | `docs/plans/代码修改与测试方案.md` 3.6 | `docs/conf.py` 及 Sphinx 扩展配置 | 未实现 | `docs/conf.py` 不存在 |
| 自动生成 API 文档 | `docs/plans/代码修改与测试方案.md` 3.6 | `docs/api/` 下存在按模块组织的 `.rst`/`.md` | 未实现 | `docs/api/` 目录为空 |

### 1.5 `src/models/roadmap.py` 中声明的 V2 远期功能

| 功能项 | 来源文档 | 预期实现 | 当前状态 | 涉及文件 |
|--------|----------|----------|----------|----------|
| ESM-3 编码器集成 | `src/models/roadmap.py` V2-01 | `ESM3Encoder` 及对应加载/推理链路 | 未实现（仅 stub） | `src/models/roadmap.py:134` |
| 质谱流式（CyTOF）实时数据支持 | `src/models/roadmap.py` V2-02 | 流式摄取 / 在线预测模块 | 未实现（仅 stub） | `src/models/roadmap.py:139` |
| 自定义 PTM 数据库接入 | `src/models/roadmap.py` V2-03 | 用户自定义 PTM 数据库加载器 | 未实现（仅 stub） | `src/models/roadmap.py:144` |
| 分布式训练（FSDP/DDP/DeepSpeed） | `src/models/roadmap.py` V2-04 | 多 GPU 分布式训练入口/辅助器 | 未实现（仅 stub） | `src/models/roadmap.py:149` |
| 图形化用户界面（GUI） | `src/models/roadmap.py` V2-05 | Web 前端或桌面 GUI | 未实现（仅 stub） | `src/models/roadmap.py:154` |

---

## 2. 仅占位或部分实现的功能项

### 2.1 模块/接口层面

| 功能项 | 来源文档 | 预期实现 | 当前状态 | 涉及文件 |
|--------|----------|----------|----------|----------|
| 扩展通路知识库程序化接口 | `AGENTS.md` 工具模块/通路知识；`docs/plans/代码修改与测试方案.md` P2-2 | 提供通路定义查询、激酶-底物查询、PTM 位点注释、通路串扰检测等 API | 仅占位：文件为大型静态数据字典集合，无查询/检测类接口 | `src/data/extended_pathway_kb.py` |
| `configs/default.yaml` 中 `features.*` 配置 | `configs/default.yaml` | `sequence_encoding`、`kmer_size`、`include_physicochemical`、`include_ptm_features` 应在主训练/推理数据流中被消费 | 部分实现：`FeatureExtractor` 读取并实现了相关特征提取逻辑，但 `PTMDataset.__getitem__` 与主训练脚本 `scripts/train.py` 未使用 `FeatureExtractor` 的输出，实际走独立的序列索引编码 | `configs/default.yaml:37-41`、`src/data/features.py`、`src/data/datasets.py` |
| `configs/production.yaml` 中 `monitoring.*` 配置 | `configs/production.yaml` | `metrics_enabled`、`metrics_port`、`health_check_interval` 应被运行时监控组件读取 | 仅占位：代码中无引用 | `configs/production.yaml:37-40` |
| API 路由聚合文件 | `AGENTS.md` 目录结构要求 `src/api/routes.py` 定义端点 | `routes.py` 直接定义 `/predict`、`/batch_predict`、`/health`、`/model_info` 等端点 | 部分实现：端点已迁移至 `src/api/routes/` 子包，`routes.py` 仅做符号转发 | `src/api/routes.py`、`src/api/routes/predictions.py`、`src/api/routes/model_info.py`、`src/api/routes/state.py` |

### 2.2 业务流程层面

| 功能项 | 来源文档 | 预期实现 | 当前状态 | 涉及文件 |
|--------|----------|----------|----------|----------|
| 基于公共 PTM 数据库的端到端数据准备流程 | `AGENTS.md` 数据库集成表 | 从 PhosphoSitePlus / dbPTM / CPLM 自动下载并构建训练数据 | 缺失入口：仅有 UniProt 加载与本地 CSV/FASTA/JSON 加载 | `src/data/loaders.py`、`scripts/prepare_data.py`、`scripts/prepare_ptm_data.py` |
| 自监督预训练 → 二分类微调流程 | `docs/plans/代码修改与测试方案.md` 3.4 | `MaskedPTMPrediction` 预训练后再用 `configs/training/finetune_binary.yaml` 微调 | 缺失入口：`finetune_binary.yaml` 存在，但自监督预训练模块不存在；`scripts/train_binary.py` 仅直接加载 CSV 训练 | `scripts/train_binary.py`、`configs/training/finetune_binary.yaml` |
| 实时质谱/流式预测流程 | `src/models/roadmap.py` V2-02 | Kafka/队列消费 + 增量 PTM 位点解码 + 在线推理 | 缺失入口：仅 `mass_spec_stream()` stub | `src/models/roadmap.py` |
| GUI/Web 交互流程 | `src/models/roadmap.py` V2-05；`docs/下一步工作规划.md` 4.3 | 序列输入、PTM 标注、预测结果可视化仪表板 | 缺失入口：仅 `launch_gui()` stub | `src/models/roadmap.py` |
| CLI 批量预测真正 batch 推理 | `README.md` 批量预测；`docs/e2e_status_report_2026_06_30.md` | 将多行输入堆叠成 batch 后统一推理 | 部分实现：`scripts/predict.py` 使用 `for row in df.iterrows()` 逐条前向；API `/batch_predict` 已实现真正 batch | `scripts/predict.py:106-149`、`src/api/routes/predictions.py:228-285` |

---

## 3. 配置文件中声明但代码未使用/未充分使用的配置项

| 配置项 | 配置文件 | 声明用途 | 当前代码状态 | 涉及文件 |
|--------|----------|----------|--------------|----------|
| `features.sequence_encoding` | `configs/default.yaml` | 控制序列特征编码方式（onehot/kmer/both） | 未被主训练/推理数据流使用；仅在 `FeatureExtractor` 内部读取 | `configs/default.yaml:38` |
| `features.kmer_size` | `configs/default.yaml` | k-mer 大小 | 同上 | `configs/default.yaml:39` |
| `features.include_physicochemical` | `configs/default.yaml` | 是否包含理化特征 | 同上 | `configs/default.yaml:40` |
| `features.include_ptm_features` | `configs/default.yaml` | 是否包含 PTM 特征 | 同上 | `configs/default.yaml:41` |
| `monitoring.metrics_enabled` | `configs/production.yaml` | 是否启用监控指标 | 代码中无引用 | `configs/production.yaml:38` |
| `monitoring.metrics_port` | `configs/production.yaml` | 监控指标端口 | 代码中无引用 | `configs/production.yaml:39` |
| `monitoring.health_check_interval` | `configs/production.yaml` | 健康检查间隔 | 代码中无引用 | `configs/production.yaml:40` |

---

## 4. 备注：与既有复核报告的差异

在本次独立复核中发现，`docs/code_status_systematic_review_report.md` 中的以下陈述与当前代码实际状态存在偏差：

- `load_from_uniprot()` 已实现（`src/data/loaders.py:149`）。
- `RegressionPredictor.forward()` 当前仅返回 `{"predictions": predictions}`（`src/models/predictors.py:72-75`）。
- `src/analysis/__init__.py` 已导出符号（`src/analysis/__init__.py:70-81`）。
- `src/data/schemas.py` 当前为 58 行，非 9 行。
- `src/models/external_tools.py`、`src/api/routes/predictions.py`、`src/api/routes/model_info.py`、`src/api/routes/state.py` 已有对应测试覆盖。
- `CHANGELOG.md`、`Dockerfile`、`docker-compose.yml`、`configs/production.yaml`、`configs/training/finetune_binary.yaml` 已存在。

本报告以上述实际代码状态为准。
---

## 附录 B：功能模块完成度详细复核（探索代理输出）

# PTM2CellNet 代码现状复核报告

**复核范围**：`src/`（data、models、training、evaluation、utils、api）、`scripts/`、`tests/`、`configs/`  
**复核日期**：2026-06-30  
**测试环境**：Python 3.12.13，pytest 9.0.3

---

## 一、总体概览

- 所有核心包均可正常导入：`src.data`、`src.models`、`src.training`、`src.evaluation`、`src.utils`、`src.api`。
- 单元测试：978 通过，2 失败，3 跳过。
- 集成测试：62 通过，3 失败，1 跳过。
- 失败集中在可选依赖缺失：`torch_geometric`（GenKI 后端）以及 `latent_vgae` 评分路径。
- 代码中存在一个关键命名冲突：`src/data/datasets.py` 中的 `PTMDataModule` 并非 Lightning DataModule，而真正的 Lightning DataModule 是 `src/data/lightning_datamodule.py` 中的 `PTMLightningDataModule`，且脚本 `scripts/train.py` 使用的是非 Lightning 版本。

---

## 二、各模块完成度评估

### 2.1 数据模块 `src/data/`

#### `loaders.py` — 完成度：中
- **已实现**：`DataLoader` 类；`load_from_csv`、`load_from_fasta`、`load_from_json`、`load_from_uniprot`、`load_sample_data`、`load_combined_data`。
- **缺失关键组件**：
  - AGENTS.md 列出的 `load_from_uniprot(self, accession_ids: List[str]) -> pd.DataFrame` 已实现，但接口签名未包含返回类型的运行时检查。
  - 未实现从 PhosphoSitePlus、dbPTM、CPLM 等数据库的加载器。
  - `load_combined_data` 仅支持简单合并，缺少按蛋白质 ID 对齐的鲁棒合并策略。
- **与需求规范的不符点**：AGENTS.md 规划的外部数据库集成点（PhosphoSitePlus、dbPTM、CPLM）均未实现。

#### `preprocess.py` — 完成度：高
- **已实现**：`DataPreprocessor` 类；`clean_sequences`、`standardize_amino_acids`、`normalize_ptm_labels`、`remove_duplicate_sequences`、`split_dataset`、`preprocess_pipeline`。
- **缺失关键组件**：无重大缺失。
- **与需求规范的不符点**：`split_dataset` 的接口签名与 AGENTS.md 基本一致，但额外增加了 `sequence_col` 参数用于同源感知分割，属于功能扩展而非缺失。

#### `features.py` — 完成度：中
- **已实现**：`FeatureExtractor` 类；`extract_sequence_features`、`extract_ptm_features`、`combine_features`、`extract_onehot_sequence`、`extract_kmer_features`、`extract_physicochemical_features`。
- **缺失关键组件**：
  - 未实现结构特征预测（AGENTS.md 中列为“结构特征预测”）。
  - 未实现特征选择 / 降维。
- **与需求规范的不符点**：`extract_ptm_features` 的签名接受 `ptm_sites_json: str` 和 `sequence_length: int`，与 AGENTS.md 中 `extract_ptm_features(self, ptm_sites: List[Dict]) -> np.ndarray` 不一致（输入为 JSON 字符串而非 List[Dict]，返回字典而非数组）。

#### `datasets.py` / `lightning_datamodule.py` — 完成度：中
- **已实现**：
  - `PTMDataset`（自定义索引编码）、`ESMTokenizedDataset`（ESM tokenizer 编码）。
  - `PTMDataModule`（非 Lightning 版本，在 `datasets.py` 中）具有 `train_dataloader`、`val_dataloader`、`test_dataloader`。
  - `PTMLightningDataModule`（Lightning 版本，在 `lightning_datamodule.py` 中）具有 `setup` 及三个 dataloader 方法。
- **缺失关键组件**：
  - `PTMDataModule`（datasets.py）缺少 `setup` 方法，不是 `L.LightningDataModule`。
  - 在线数据增强（`src/data/augmentation.py` 存在 `SequenceAugmenter`、`PTMAugmenter`，但 `PTMDataset` 中未调用）。
- **与需求规范的不符点**：AGENTS.md 中 `PTMDataModule` 被定义为 `pl.LightningDataModule` 并需实现 `setup`。当前存在两个同名/不同类的实现，形成命名冲突；`scripts/train.py` 导入的是非 Lightning 版本，导致与 Lightning 生态不兼容。

#### `dataset_base.py` — 完成度：高
- **已实现**：`PTMDatasetBase` 基类、标签编码、PTM 解析、统计信息、`compute_class_weights`。
- **缺失关键组件**：无。

#### `validation.py` / `augmentation.py` / `multitask_dataset.py` / `ptm_site_dataset.py` — 完成度：中
- **已实现**：数据验证器、缓存、增强器、多任务/PTM 位点专用数据集。
- **缺失关键组件**：增强器未接入主 `PTMDataset`；多任务/PTM 位点数据集与主流程的整合度低。

---

### 2.2 模型模块 `src/models/`

#### `encoders.py` — 完成度：高
- **已实现**：`SequenceEncoder` 基类、`CNNEncoder`、`TransformerEncoder`、`LSTMEncoder`、`GRUEncoder`、`PositionalEncoding`。
- **缺失关键组件**：无。
- **与需求规范的不符点**：接口与 AGENTS.md 一致。

#### `ptm_modules.py` — 完成度：高
- **已实现**：`PTMEmbedding`、`PTMAttention`、`PTMModule`、`GatedPTMFusion`。
- **缺失关键组件**：无。
- **与需求规范的不符点**：`PTMAttention.forward` 额外支持 `ptm_mask` 参数，属于扩展。

#### `predictors.py` — 完成度：高
- **已实现**：`CellStatePredictor` 基类、`ClassificationPredictor`、`RegressionPredictor`。
- **缺失关键组件**：无。
- **与需求规范的不符点**：无。

#### `architectures.py` — 完成度：高
- **已实现**：`PTM2CellNet`、`PTM2CellNetLarge`，支持 CNN/Transformer/LSTM/GRU/Mamba/ESM-2/ProtBERT/ProtT5 编码器；`from_config` 类方法；DAVF 分支集成。
- **缺失关键组件**：无。
- **与需求规范的不符点**：AGENTS.md 中 `PTM2CellNet.__init__` 示意接收 `config: Dict`，实际为显式参数 + `from_config` 工厂方法，属于实现方式差异。

#### `pooling.py` / `multitask.py` / `ensemble.py` — 完成度：高
- **已实现**：多种池化层、`MultiTaskPredictor`/`HierarchicalMultiTaskPredictor`、`PTM2CellNetEnsemble`。
- **缺失关键组件**：无。

#### `pretrained_encoders.py` / `mamba_encoder.py` — 完成度：高
- **已实现**：`ESM2Encoder`、`ProtBERTEncoder`、`ProtT5Encoder` 及 `MambaEncoder`。
- **缺失关键组件**：`mamba-ssm` 为可选依赖；当未安装时使用自定义 `SelectiveSSM`。
- **与需求规范的不符点**：无。

#### `external_tools.py` — 完成度：中
- **已实现**：`AlphaFoldClient`、`BLASTClient`、`ClustalWClient`、`PSIPREDClient`，均带 API/本地工具回退。
- **缺失关键组件**：部分功能依赖外部工具/API 可用性，本地无安装时降级为简单实现。
- **与需求规范的不符点**：AGENTS.md 期望“本地安装 / API”方式，当前实现更偏向 API 调用 + 内置回退。

#### DAVF / scVI / SignalingNetwork / Variant / PTM Direction 相关模块 — 完成度：中
- **已实现**：`PTMDirectionMapper`、`DAVFInferenceModule`、`DeltaPredictor`、`ScVIAdapter`、`SignalingNetworkMapper`、`VariantEffectWorkflow` 等。
- **缺失关键组件**：
  - `ScVIAdapter` 依赖 `scvi-tools`（requirements.txt 已列出，但可能未安装）。
  - DAVF 分支需要预训练检查点 `checkpoints/latent_davf_ibd_norman/best_model.pt` 及 scVI 模型路径。
- **与需求规范的不符点**：这些属于 AGENTS.md 未列出的后续阶段功能，不属于 v1.0 核心契约，但代码中已大量存在。

---

### 2.3 训练模块 `src/training/`

#### `trainers.py` — 完成度：高
- **已实现**：`Trainer` 类；`compile`、`fit`、`_train_epoch`、`_val_epoch`、回调机制。
- **缺失关键组件**：`validate(self, model, dataloader) -> Dict[str, float]` 方法未实现（AGENTS.md 列出）。
- **与需求规范的不符点**：AGENTS.md 中的 `Trainer.validate` 接口缺失；当前验证通过 `_val_epoch` 在 `fit` 内部完成。

#### `losses.py` — 完成度：高
- **已实现**：`FocalLoss`、`DiceLoss`、`MultiTaskLoss`。
- **缺失关键组件**：无。
- **与需求规范的不符点**：无。

#### `optimizers.py` — 完成度：高
- **已实现**：`configure_optimizer`；支持 Adam/AdamW/SGD、Cosine/Plateau/Step 调度器。
- **缺失关键组件**：无。
- **与需求规范的不符点**：无。

#### `callbacks.py` — 完成度：高
- **已实现**：`Callback` 基类、`ModelCheckpoint`、`EarlyStopping`、`LearningRateMonitor`、`ProgressBarCallback`。
- **缺失关键组件**：无。
- **与需求规范的不符点**：AGENTS.md 中 `ModelCheckpoint` 和 `EarlyStopping` 被标注为 `pl.Callback`，实际为自定义 `Callback` 子类，不是 PyTorch Lightning 的 `Callback`。

#### `lightning_module.py` / `logging_config.py` / `peft_config.py` — 完成度：高
- **已实现**：`PTM2CellNetLightning`、默认 logger 工厂、LoRA/PEFT 配置。
- **缺失关键组件**：无。

---

### 2.4 评估模块 `src/evaluation/`

#### `metrics.py` — 完成度：高
- **已实现**：分类、回归、排序、MCC、AUPR、per-PTM 类型指标等。
- **缺失关键组件**：无。
- **与需求规范的不符点**：无。

#### `evaluators.py` — 完成度：高
- **已实现**：`Evaluator` 类；`evaluate`、`cross_validate`。
- **缺失关键组件**：置信区间计算在 `metrics.py` 中实现，但 `Evaluator` 未直接集成。
- **与需求规范的不符点**：无重大不符。

#### `visualization.py` — 完成度：高
- **已实现**：`plot_roc_curve`、`plot_pr_curve`、`plot_confusion_matrix`、`plot_training_curves`、`plot_feature_importance`、`plot_attention_heatmap`。
- **缺失关键组件**：无。

#### `explainers.py` — 完成度：中
- **已实现**：`LeaveOnePTMOutScorer`、`TwoStageExplanationPipeline`。
- **缺失关键组件**：GenKI 后端依赖 `torch_geometric`，当前环境缺失，导致相关集成测试失败。

---

### 2.5 工具模块 `src/utils/`

#### `config.py` — 完成度：高
- **已实现**：`Config` 类；`from_yaml`、`get`（支持点号）、`set`、`to_dict`、`save`。
- **缺失关键组件**：无。
- **与需求规范的不符点**：无。

#### `logging.py` — 完成度：高
- **已实现**：`setup_logger`。
- **缺失关键组件**：无。

#### `io.py` — 完成度：高
- **已实现**：`save_pickle`、`load_pickle`、`save_model`、`load_model`、JSON/CSV/NumPy 读写。
- **缺失关键组件**：HDF5 读写未实现。
- **与需求规范的不符点**：AGENTS.md 提到 HDF5 支持，当前未实现。

#### `helpers.py` — 完成度：高
- **已实现**：`validate_sequence`、`validate_ptm_site`、`clean_sequence` 等。
- **缺失关键组件**：无。

---

### 2.6 API 模块 `src/api/`

#### `app.py` — 完成度：高
- **已实现**：`create_app` 工厂函数；CORS、生命周期管理、自动模型初始化。
- **缺失关键组件**：无。

#### `routes/` / `routes.py` — 完成度：高
- **已实现**：
  - `POST /predict`
  - `POST /predict/batch` 与 `POST /batch_predict`（别名）
  - `GET /health`
  - `GET /model/info` 与 `GET /model_info`（别名）
  - `POST /predict/variant`（扩展功能）
- **缺失关键组件**：无。
- **与需求规范的不符点**：AGENTS.md 中列出的 `POST /batch_predict` 当前以 `/predict/batch` 为主路径，`/batch_predict` 为别名。

#### `schemas.py` — 完成度：高
- **已实现**：`PTMSite`、`PredictionRequest`、`BatchPredictionRequest`、`PredictionResponse`、`BatchPredictionResponse`、`HealthResponse`、`ModelInfoResponse` 及变异预测相关模型。
- **缺失关键组件**：无。

---

## 三、端到端流程贯通性

| 环节 | 状态 | 说明 |
|---|---|---|
| 原始数据 → DataLoader | ✅ 贯通 | `DataLoader.load_from_csv` / `load_sample_data` 可用。 |
| DataLoader → Preprocessor | ✅ 贯通 | `DataPreprocessor.preprocess_pipeline` 可接收 DataFrame 并输出 train/val/test。 |
| Preprocessor → FeatureExtractor | ⚠️ 部分贯通 | `FeatureExtractor` 独立存在，但 `PTMDataset` 默认使用内部 `_encode_sequence`，未显式调用 `FeatureExtractor.extract_sequence_features`。 |
| FeatureExtractor → Dataset | ⚠️ 部分贯通 | `PTMDataset` 接收 `feature_extractor` 参数但主要用于配置读取；特征提取与数据集编码路径分离。 |
| Dataset → DataModule | ⚠️ 命名冲突 | `PTMDataModule` 在 `datasets.py` 中为非 Lightning 版本；Lightning 版本为 `PTMLightningDataModule`。`scripts/train.py` 使用非 Lightning 版本，`scripts/train_lightning.py` 使用 Lightning 版本。 |
| DataModule → Model | ✅ 贯通 | `PTM2CellNet.from_config` 可正常创建模型。 |
| Model → Trainer | ✅ 贯通 | `Trainer` 和 `PTM2CellNetLightning` 均可接收模型。 |
| Trainer → Evaluator | ✅ 贯通 | `Evaluator.evaluate` 可接收 DataLoader 并输出指标。 |
| Evaluator → Visualization | ✅ 贯通 | 可视化函数可直接使用评估结果。 |

**关键断裂点**：
1. `PTMDataModule` 命名冲突导致 AGENTS.md 中定义的 Lightning DataModule 契约不唯一。
2. `FeatureExtractor` 与 `PTMDataset` 的编码逻辑存在重复（one-hot/PTM 编码各实现一份），未形成单一特征提取管道。
3. 数据增强模块未接入主数据集。
4. 结构特征、HDF5 IO、PhosphoSitePlus/dbPTM/CPLM 加载器等 AGENTS.md 规划功能未实现。

---

## 四、测试与配置现状

### 4.1 测试
- **单元测试**：覆盖 data、models、training、evaluation、api、integration/genki、analysis 等模块；核心模块测试通过率高。
- **失败测试**：
  - `tests/unit/integration/test_genki_adapter.py::TestBackendDetection::test_get_backend_info_genki_source`
  - `tests/unit/integration/test_genki_reference_data.py::TestReferenceDataErrors::test_load_reference_data_genki_source`
  - `tests/integration/test_soft_perturbation_pipeline.py`（2 个）
  - `tests/integration/test_two_stage_pipeline.py::test_two_stage_cli_accepts_genki_source_backend`
- **失败原因**：`torch_geometric` 未安装；`latent_vgae` 评分路径依赖未满足。

### 4.2 配置
- `configs/default.yaml`、`configs/lightning.yaml` 等存在，覆盖默认训练、Lightning 训练、预训练模型、DAVF 集成等场景。
- `requirements.txt` 已列出 `scvi-tools`、`mamba-ssm`，但未列出 `torch_geometric`，而 GenKI 后端需要它。

---

## 五、与 AGENTS.md 接口契约的主要偏差汇总

| 契约项 | 现状 | 偏差 |
|---|---|---|
| `DataLoader` 外部数据库加载 | 仅 UniProt | PhosphoSitePlus、dbPTM、CPLM 未实现 |
| `FeatureExtractor.extract_ptm_features` | 输入 JSON 字符串，返回字典 | 与 AGENTS.md 的 `List[Dict] -> np.ndarray` 不一致 |
| `PTMDataModule` 为 Lightning DataModule | 存在非 Lightning 版本与 Lightning 版本两个类，命名冲突 | 契约不唯一 |
| `Trainer.validate` | 未实现 | 缺失 |
| `ModelCheckpoint` / `EarlyStopping` | 自定义 Callback 子类 | 不是 `pl.Callback` |
| IO 支持 HDF5 | 未实现 | 缺失 |
| 数据增强接入主 Dataset | 未接入 | 缺失 |
| 结构特征预测 | 未实现 | 缺失 |

---

**复核结论**：核心分类/回归训练与预测流程已实现并可运行，测试覆盖较好。主要问题在于 `PTMDataModule` 命名冲突、部分 AGENTS.md 规划功能（外部数据库、HDF5、结构特征、数据增强接入）未实现，以及可选依赖（`torch_geometric`、`scvi-tools` 等）的完整性不足。
---

## 附录 C：未报告技术债详细复核（探索代理输出）

# PTM2CellNet 未报告技术债复核报告

**复核范围**：`src/`、`scripts/`、`tests/`、`configs/`、`requirements.txt`、`setup.py`  
**复核原则**：先阅读 `docs/` 下已有报告，剔除已明确提及的问题，仅列出未被已有报告覆盖的技术债。  
**已有报告已审阅**：`benchmark_report.md`、`code_status_systematic_review_report.md`、`CodeReview修复报告.md`、`GPU训练修复报告.md`、`Mamba验证报告.md`、`e2e_status_report_2026_06_30.md`、`PTM2CellNet技术深度分析报告.md`、`performance_benchmark.md`。

---

## 1. 代码质量问题

| 类别 | 位置 | 问题描述 | 严重程度 |
|---|---|---|---|
| 重复/近似代码块 | `src/models/ptm_site_predictor.py:99`、`src/models/multitask_ptm.py:147`、`src/models/biperturb.py:433`、`src/models/davf.py:495`、`src/models/latent_davf.py:226` | 多处存在几乎一致的 `_init_weights` 实现，未抽取到公共工具函数。 | 低 |
| 重复/近似代码块 | `src/models/delta_predictor.py:154-155`、`src/models/delta_predictor.py:222`、`src/models/davf.py:650`、`src/models/latent_davf.py:348`、`src/models/latent_davf.py:411` | 多处存在相同的条件判断/数据整理片段，未复用。 | 低 |
| 重复/近似代码块 | `src/models/architectures.py:118/126/134/501/509` | `PTM2CellNet` 与 `PTM2CellNetLarge` 中编码器选择逻辑重复出现，未抽象为共享函数。 | 中 |
| 重复/近似代码块 | `src/integration/genki_adapter.py:40-70`、`src/integration/genki/perturbation.py:36-66`、`src/integration/genki/reference_data.py:27-57`、`src/integration/genki/significance.py:31-61` | GenKI 相关模块顶部存在大段重复的依赖探测与 `_BACKEND` 注册代码。 | 中 |
| 行尾格式不一致 | `scripts/predict.py`、`scripts/evaluate.py`、`scripts/train.py`、`scripts/data_statistics.py`、`scripts/train_lightning.py`、`scripts/train_pretrained.py` | 多个脚本文件使用 CRLF 或混合 CRLF/LF 行尾，与项目其他文件不一致，可能导致 diff 噪音与跨平台问题。 | 低 |
| 硬编码路径 | `scripts/predict.py:21`、`scripts/evaluate.py:22`、`scripts/train.py:22`、`scripts/train_lightning.py`、`scripts/train_pretrained.py` 等 | 大量脚本默认参数硬编码 `outputs/models/best_model.pt`、`configs/default.yaml`、`outputs/results` 等路径，未统一从配置读取。 | 中 |
| 魔法数字 | `src/api/routes/predictions.py:204`、`src/api/routes/predictions.py:349` | 通路活性阈值 `0.5` 硬编码用于判断 confidence="high" 或 "medium"，未参数化。 | 低 |
| 魔法数字 | `src/api/routes/predictions.py:194`、`src/api/routes/predictions.py:356` | `delta_prob` 默认 `1.0`、`confidence = min(max_delta * 2, 1.0)` 中的 `2` 与 `1.0` 未说明来源。 | 低 |
| 未处理异常/静默吞错 | `scripts/integrate_data_v2.py:206-207` | `except Exception: pass` 捕获批量解析 UniProt 响应时的所有异常，可能隐藏网络或格式错误。 | 中 |
| 未处理异常/静默吞错 | `scripts/finetune_davf.py:292-293` | `except Exception: pass` 在计算 AUC-ROC 时静默吞掉所有错误。 | 中 |
| 未处理异常 | `scripts/ensemble_predict.py:148-149` | `predict_voting` 中捕获 `Exception` 仅记录 warning，失败模型被跳过但无返回错误。 | 中 |
| 函数嵌套定义 | `scripts/train.py:169-176` | `_sanitize_nans` 作为 `main()` 内部嵌套函数定义，降低可测试性与可读性。 | 低 |
| 输入验证缺失 | `scripts/ensemble_predict.py:112` | `ESM2Wrapper.forward` 直接返回零张量，是桩实现，未对真实输入做有效推理。 | 中 |
| 命名/语义不一致 | `src/models/architectures.py:681` | 注释写 `[B, embed_dim + 128]`，但 `128` 与上游 `feature_dim` 默认值可能不一致，硬编码数字未引用配置。 | 低 |
| 类型注解缺失 | `scripts/` 下多个脚本 | `parse_args()`、`main()` 及辅助函数普遍缺少返回类型与参数类型注解。 | 低 |

---

## 2. 架构缺陷

| 类别 | 位置 | 问题描述 | 严重程度 |
|---|---|---|---|
| 模块导入风格不一致 | `scripts/ensemble_predict.py:11`、`scripts/batch_predict.py:11`、`scripts/prepare_ptm_data.py:19` 等 | 多数脚本顶部直接 `sys.path.insert`，而部分脚本使用 `_ensure_project_root()` 辅助函数；未统一项目根目录注入方式。 | 低 |
| 模块导入风格不一致 | `src/api/routes.py:3`、`src/data/preprocess.py:13`、`src/data/features.py:12`、`src/models/davf.py:19-24` 等 | 部分模块使用 `from src.xxx import ...` 绝对导入，破坏包内相对导入约定。 | 低 |
| 职责不清/重复设备判断 | `scripts/evaluate.py:85`、`scripts/predict.py:61`、`scripts/train_multitask_ptm.py`、`scripts/batch_predict.py` 等 | 各脚本独立重复 `args.device or ("cuda" if torch.cuda.is_available() else "cpu")`，未复用 `src/utils/io.py` 或统一工具。 | 低 |
| 配置与实现脱节 | `configs/production.yaml:10`、`configs/davf_integration.yaml:7`、`src/models/architectures.py:203/578`、`src/models/davf_inference.py:57` | 默认 checkpoint 路径 `outputs/models/best_model.pt`、`checkpoints/latent_davf_ibd_norman/best_model.pt` 在多处配置与代码中重复硬编码，未集中管理。 | 中 |
| 配置与实现脱节 | `configs/default.yaml` vs `src/api/app.py:25-30` | API 启动时默认细胞状态列表 `proliferation,differentiation,apoptosis,quiescence` 在代码中硬编码，未从生产配置读取。 | 低 |
| 职责不清 | `src/api/routes/predictions.py:32-94` | `preprocess_request()` 同时承担序列验证、PTM 解析、张量构造与全局状态访问，单一函数职责过重。 | 中 |
| 抽象基类缺少显式说明 | `src/models/encoders.py:17`、`src/models/predictors.py:26` | 抽象接口方法标注 `# pragma: no cover - interface`，但文档与类型层面未使用 `ABC`/`abstractmethod` 强制约束。 | 低 |

---

## 3. 性能瓶颈

| 类别 | 位置 | 问题描述 | 严重程度 |
|---|---|---|---|
| 批量推理未真正 batch 化 | `scripts/batch_predict.py` | 类名虽为 `BatchPTMPredictor`，但内部对每条序列独立 tokenize/前向，未将多行堆叠成 batch。 | 高 |
| 循环内设备同步 | `scripts/ensemble_predict.py:141-147` | 对每模型、每 batch 在循环中调用 `.cpu().numpy()`，形成 CPU/GPU 同步点。 | 中 |
| 低效的 Python 级循环 | `src/data/features.py:98`、`src/data/homology_splitter.py:37-38` | k-mer 提取使用 Python `range(len(...))` 循环，未使用向量化或 `zip`/`itertools`。 | 低 |
| 低效的 Python 级循环 | `src/evaluation/visualization.py:244-245` | 使用 `range(len(...))` 索引循环绘制条形图，可用向量化 API 替代。 | 低 |
| 重复 I/O 加载 | `scripts/validate_cptac.py:231-250` | 对每个 PTM 类型分别扫描目录并加载 checkpoint，未缓存模型句柄。 | 低 |
| 未关闭/未复用资源 | `src/data/loaders.py:165-178` | `load_from_uniprot` 回退逐个请求时未使用 session/连接复用。 | 低 |

---

## 4. 安全隐患

| 类别 | 位置 | 问题描述 | 严重程度 |
|---|---|---|---|
| 不安全的反序列化 | `scripts/batch_predict.py:122`、`scripts/batch_predict.py:147` | `torch.load(self.model_path, map_location=self.device)` 未指定 `weights_only`，使用默认 pickle 反序列化。 | 高 |
| 不安全的反序列化 | `scripts/validate_cptac.py:250` | `torch.load(best_model, map_location='cpu')` 未指定 `weights_only`。 | 高 |
| 不安全的反序列化 | `scripts/train_multitask_ptm.py:291` | `torch.load(checkpoint_dir / 'best_model.ckpt')` 未指定 `weights_only`。 | 高 |
| 不安全的反序列化 | `scripts/ensemble_predict.py:116` | `torch.load(path, map_location=self.device)` 未指定 `weights_only`（仅同文件第 82 行显式传了 `False`）。 | 高 |
| 不安全的反序列化 | `tests/unit/test_training_inference_consistency.py:82/144/584` | 测试代码中 `torch.load(...)` 未指定 `weights_only`。 | 中 |
| 输入未经验证 | `scripts/ensemble_predict.py:289-355` | CLI 参数 `--models`、`--types`、`--weights` 长度不一致时仅按索引取默认值，未显式校验对齐关系。 | 低 |
| 输入未经验证 | `scripts/predict_variant_effect.py:56-57` | `--variant` 格式说明为 `UniProtID:Position:RefAA:AltAA`，但 `README.md:98` 示例为 `--variant "BRAF_V600E"`，格式不一致且无输入格式校验。 | 中 |
| 宽异常捕获 | `scripts/integrate_data_v2.py:206-207` | `except Exception: pass` 可能隐藏网络/解析错误。 | 中 |
| 宽异常捕获 | `scripts/finetune_davf.py:292-293` | `except Exception: pass` 可能隐藏指标计算错误。 | 中 |

---

## 5. 文档缺失

| 类别 | 位置 | 问题描述 | 严重程度 |
|---|---|---|---|
| 公共函数/类缺少 docstring | `src/integration/ptm_virtual_perturbation.py:10/16`、`src/integration/contracts.py:8/20/30`、`src/integration/genki_adapter.py:29/108/117/121/124/127/136/139/142/145`、`src/integration/genki_reports.py:186` 等 | 大量公共类与函数缺少 docstring。 | 中 |
| 公共函数/类缺少 docstring | `src/training/peft_config.py:34/35/38/41`、`src/training/ptm_site_lightning.py:69/122/125/128`、`src/evaluation/explainers.py:23/33/51/63/90/110/137/144/176/190/265`、`src/evaluation/visualization.py:23` 等 | 公共 API 缺少文档字符串。 | 中 |
| 公共函数/类缺少 docstring | `src/models/predictors.py:12/26/33/45/60/72`、`src/models/encoders.py:17/24/37/45/52/63/96/115/138/148/171`、`src/models/ptm_modules.py:15/22/33/38/64/86/117/124`、`src/models/davf.py:74/139/197/298/455/502/535/776/872/925` 等 | 核心模型类与函数缺少 docstring。 | 中 |
| 公共函数/类缺少 docstring | `src/data/validation.py:26/189`、`src/data/dataset_base.py:23/119/256`、`src/data/multitask_dataset.py:110/113/248/257/266/296/300/303/317`、`src/data/ptm_site_dataset.py:85/256/266/275` 等 | 数据模块公共接口缺少文档字符串。 | 中 |
| README 与代码不符 | `README.md:98` | 示例 `python scripts/predict_variant_effect.py --variant "BRAF_V600E"` 与脚本实际要求的 `UniProtID:Position:RefAA:AltAA` 格式不一致。 | 中 |
| 脚本缺少文档 | `scripts/batch_predict.py:41/64/69/118/132/150/151/165/185/472`、`scripts/train_binary.py:40/66/84/183`、`scripts/train_multitask_ptm.py:34/90/150` 等 | 脚本中的类与函数缺少 docstring 或参数说明。 | 低 |

---

## 6. 测试覆盖率不足

| 类别 | 位置 | 问题描述 | 严重程度 |
|---|---|---|---|
| 占位测试未实现 | `tests/unit/test_api_routes.py:309` | `TestWithRealWeights.test_predict_with_production_model` 仅包含 `pass` 占位，真实生产检查点集成测试缺失。 | 中 |
| 脚本测试覆盖不足 | `scripts/ensemble_predict.py`、`scripts/batch_predict.py`、`scripts/validate_cptac.py`、`scripts/train_multitask_ptm.py`、`scripts/integrate_data_v2.py`、`scripts/finetune_davf.py` | 大量脚本无对应单元/集成测试，关键 CLI 路径未覆盖。 | 高 |
| 条件分支未覆盖 | `src/api/routes/predictions.py` | `/predict/variant` 与 `/batch_predict` 的错误分支、通路分析失败分支缺乏专门测试。 | 中 |
| 环境跳过测试较多 | `tests/unit/test_models.py`、`tests/unit/test_scvi_adapter.py`、`tests/unit/training/test_peft_config.py`、`tests/unit/test_training_inference_consistency.py` 等 | 存在大量 `@pytest.mark.skipif` 与运行时 `pytest.skip`，可选依赖缺失时关键能力未在 CI 中验证。 | 中 |

---

## 7. 依赖管理问题

| 类别 | 位置 | 问题描述 | 严重程度 |
|---|---|---|---|
| 可选依赖未声明/未隔离 | `scripts/ensemble_predict.py:90`、`scripts/batch_predict.py:134` | 直接 `import esm`（fair-esm），但未在 `requirements.txt` 或 `setup.py` 中声明。 | 中 |
| 可选依赖未声明/未隔离 | `scripts/validate_cptac.py` | 可能依赖 CPTAC/TCGA 相关库，脚本内未做依赖可用性检查与降级。 | 低 |
| 依赖版本约束过宽 | `requirements.txt:7`、`requirements.txt:8`、`requirements.txt:9` | `torch>=1.10.0`、`torchvision>=0.11.0`、`lightning>=2.5.0` 版本下界与新版 Lightning 对 PyTorch 的要求存在潜在冲突，且无锁定文件。 | 中 |
| 脚本依赖未统一 | 多个 `scripts/*.py` | 各脚本自行 `sys.path.insert` 并在函数内延迟导入，未通过 `setup.py` 安装后的包名导入，导致依赖关系不透明。 | 低 |

---

**说明**：本报告仅客观罗列现状，未包含修改建议。所列问题均已通过源码阅读、AST 扫描、静态检查或测试运行确认，且未在已有 `docs/` 报告中明确提及。