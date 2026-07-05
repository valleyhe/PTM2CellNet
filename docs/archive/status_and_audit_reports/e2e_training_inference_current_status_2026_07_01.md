# PTM2CellNet 当前 E2E 训练与推理现状审计

审计日期: 2026-07-01  
审计范围: 当前工作区代码、配置、CLI 入口、API 初始化路径、相关测试与最小端到端冒烟运行。  
结论: 当前项目只能证明“部分入口可以跑通”，尚未满足可靠的 E2E 训练和推理要求。阻塞点主要集中在 checkpoint 格式不统一、标签映射未持久化、API 权重加载过于宽松、批量推理丢失 PTM 输入，以及缺少真正覆盖训练产物到推理入口的集成测试。

## 1. 总体判断

### 已具备的能力

1. 原生训练入口 `scripts/train.py` 可以完成最小训练、保存 `best_model.pt`、保存同名配置、生成测试指标。
2. `scripts/predict.py` 可以加载 `scripts/train.py` 产出的裸 `state_dict` checkpoint，并完成单序列推理。
3. Lightning 组件级训练链路可运行，相关组件测试通过。
4. 现有配置解析、数据加载、预处理、Dataset、模型前向、训练循环、指标评估等基础模块基本存在。

### 尚未达标的原因

1. 原生训练与推理虽能运行，但训练标签顺序没有被保存，推理使用硬编码标签顺序，可能把模型类别索引解码成错误细胞状态。
2. Lightning 训练入口产出 `.ckpt`，当前 CLI 推理无法加载该格式。
3. API `/initialize` 和启动自动初始化使用 `strict=False`，在 checkpoint 不匹配时仍可能返回加载成功，导致生产推理使用随机初始化或部分初始化模型。
4. 批量推理 CSV 路径忽略输入中的 `ptm_sites`，实际只用序列推理，和 PTM2CellNet 的任务定义不一致。
5. 现有测试覆盖了组件、help、轻量 forward，但没有覆盖“训练 CLI 产物 -> 推理 CLI/API -> 标签语义正确”的真实 E2E 合约。

## 2. 取证摘要

### 2.1 通过的验证

执行命令:

```bash
pytest -q tests/unit/test_scripts.py tests/unit/test_training_inference_consistency.py tests/integration/test_lightning_pipeline.py tests/integration/test_train_lightning_script.py
```

结果:

```text
42 passed, 2 skipped
```

执行原生 E2E 冒烟:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data data/raw/sample_data.csv \
  --output /tmp/ptm2cellnet_e2e_audit \
  --epochs 1 \
  --batch_size 8

python scripts/predict.py \
  --model /tmp/ptm2cellnet_e2e_audit/models/best_model.pt \
  --sequence ACDEFGHIKLMNPQRSTVWY \
  --device cpu
```

结果: 训练完成并生成 `/tmp/ptm2cellnet_e2e_audit/models/best_model.pt`、`best_model.config.yaml`、测试指标；推理完成并输出概率。

### 2.2 失败或有风险的验证

Lightning 训练可以生成 `.ckpt`:

```bash
python scripts/train_lightning.py \
  --config /tmp/ptm2cellnet_lightning_audit.yaml \
  --data data/raw/sample_data.csv \
  --max-epochs 1 \
  --batch-size 16 \
  --precision 32
```

但使用当前推理 CLI 加载 Lightning checkpoint 失败:

```bash
python scripts/predict.py \
  --model /tmp/ptm2cellnet_lightning_audit/models/audit-lightning.ckpt \
  --config /tmp/ptm2cellnet_lightning_audit.yaml \
  --sequence ACDEFGHIKLMNPQRSTVWY \
  --device cpu
```

关键错误:

```text
Missing key(s) in state_dict: "encoder.embedding.weight", ...
Unexpected key(s) in state_dict: "epoch", "global_step", "pytorch-lightning_version", "state_dict", ...
```

API 初始化同一个 Lightning checkpoint 时返回 `success`，但日志显示缺失和多余键:

```text
initialize: 缺失键 ['encoder.embedding.weight', ...]
initialize: 多余键 ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops']
status= success
```

这说明 API 当前可能接受了没有正确加载模型权重的 checkpoint。

## 3. 阻塞性问题与严重问题

### P0-1: 训练标签映射未持久化，推理标签可能错误

证据:

- `PTMDatasetBase` 用 `sorted(self.df["cell_state"].unique())` 生成训练标签顺序: `src/data/dataset_base.py:35-38`。
- `scripts/train.py` 只设置 `model.num_classes` 并保存配置，没有保存 `data.cell_states`: `scripts/train.py:114-116`、`scripts/train.py:163-166`。
- `scripts/predict.py` 在缺少 `data.cell_states` 时使用硬编码顺序 `["proliferation", "differentiation", "apoptosis", "quiescence"]`: `scripts/predict.py:119-121`。
- 实测训练标签顺序为 `['apoptosis', 'differentiation', 'proliferation', 'quiescence']`，保存的 `best_model.config.yaml` 中 `data.cell_states = None`。

影响:

原生训练/推理即使命令成功，模型输出类别索引也可能映射到错误细胞状态。例如训练时索引 0 是 `apoptosis`，推理 fallback 中索引 0 是 `proliferation`。

解决策略:

1. 在训练完成前将 `cell_states` 写入配置:
   - `config.set("data.cell_states", cell_states)`
   - 同时保存 `label_to_idx` 或 `idx_to_label` 到 checkpoint metadata，避免只依赖 YAML。
2. 推理入口禁止静默 fallback 到硬编码顺序:
   - 若 checkpoint 配套配置没有 `data.cell_states`，应报错并提示用户提供标签映射。
   - 仅允许 demo 模型显式启用 fallback。
3. 增加 E2E 测试:
   - 构造非硬编码顺序标签。
   - 训练 1 epoch 保存产物。
   - 加载推理并断言 `cell_states` 与训练 `datamodule.get_labels()` 完全一致。

### P0-2: API checkpoint 加载使用 `strict=False`，会把坏 checkpoint 当成功加载

证据:

- API 热加载路径用 `model.load_state_dict(state_dict, strict=False)`: `src/api/routes/initialize.py:105-115`。
- API 启动自动初始化也用 `strict=False`: `src/api/app.py:50-53`。
- 实测传入 Lightning `.ckpt` 时 API 返回 `status=success`，但日志显示模型键缺失、checkpoint 外层键多余。

影响:

生产服务可能在 checkpoint 不匹配时仍返回健康或加载成功，后续预测来自随机初始化模型或部分加载模型。这比显式失败更危险。

解决策略:

1. 新增统一 checkpoint 解析函数，例如 `extract_model_state_dict(checkpoint)`:
   - 裸 `state_dict`: 直接返回。
   - Lightning `.ckpt`: 若包含 `state_dict`，剥离 `model.` 前缀后返回。
   - 旧格式: 支持 `model_state_dict`。
2. 默认使用严格加载:
   - `missing` 和 `unexpected` 非空时直接失败。
   - 仅对明确声明的兼容迁移路径允许非严格加载，并返回 warning 状态而不是 success。
3. API 初始化返回中加入:
   - `loaded_format`
   - `missing_keys_count`
   - `unexpected_keys_count`
   - `loaded_parameter_ratio`
4. 增加负向测试:
   - 给 API `/initialize` 传入 Lightning 原始 `.ckpt`，在未实现格式转换前必须失败。
   - 实现转换后必须验证加载后参数覆盖率接近 100%。

### P0-3: Lightning 训练产物无法被当前推理 CLI 加载

证据:

- `scripts/train_lightning.py` 使用 Lightning `ModelCheckpoint` 保存 `.ckpt`: `scripts/train_lightning.py:171-186`。
- `scripts/predict.py` 通过 `load_model(model, args.model)` 加载裸模型 state_dict: `scripts/predict.py:123-133`。
- `src/utils/io.py` 的 `load_model` 直接 `model.load_state_dict(safe_torch_load(...))`，不解析 Lightning 外层结构。

影响:

项目存在两条训练入口，但只有原生训练产物能被当前推理 CLI 可靠消费。用户使用 Lightning 训练后无法直接进行推理。

解决策略:

1. 统一训练产物格式:
   - 首选: 所有训练入口都额外导出 `best_model.pt` 裸 `state_dict` 与 `best_model.config.yaml`。
   - 保留 `.ckpt` 用于恢复训练。
2. 或统一推理加载器:
   - `load_model` 支持 `.pt`、`.pth`、`.ckpt`。
   - 自动识别 `state_dict`、`model_state_dict`、裸字典。
   - 对 Lightning 键名前缀 `model.` 做规范化。
3. `scripts/train_lightning.py` 在训练结束后保存:
   - 最佳裸模型权重。
   - 同名配置。
   - 标签映射。
4. 增加 CLI E2E 测试:
   - `train_lightning.py --max-epochs 1`
   - `predict.py --model <lightning-exported-best_model.pt>`
   - 断言退出码为 0 且输出类别映射正确。

### P1-1: 批量推理忽略 CSV 中的 PTM 位点

证据:

- `scripts/predict.py` 批量模式读取每行 `sequence` 后创建 `PredictionRequest(..., ptm_sites=[])`: `scripts/predict.py:187-193`。

影响:

批量推理不会使用输入文件中的 PTM 信息，模型退化为“序列-only”推理。这与项目目标“蛋白质序列 + PTM 信息 -> 细胞状态”不一致。

解决策略:

1. 批量 CSV 支持 `ptm_sites` JSON 列，复用 `DataPreprocessor.normalize_ptm_labels` 或 Dataset 的 PTM 解析逻辑。
2. 对非法 PTM JSON 和越界位点给出行级错误，输出 `error` 列或 fail-fast 模式。
3. 增加测试:
   - 同一 sequence、不同 ptm_sites，预处理后的 `ptm_mask`/`ptm_types` 必须不同。
   - 批量推理 helper 必须把 CSV 的 PTM 信息传入模型。

### P1-2: 单样本 CLI 无法传入 PTM 位点，也不保存单样本输出

证据:

- 单样本模式只接受 `--sequence`，构造 `ptm_sites=[]`: `scripts/predict.py:139-144`。
- 单样本结果只写日志，没有写到 `--output`: `scripts/predict.py:161-174`。

影响:

用户无法通过 CLI 做真实 PTM-aware 单样本预测，也难以在流水线中消费结果。

解决策略:

1. 增加 `--ptm-sites` 参数，接受 JSON 字符串或 JSON 文件。
2. 单样本模式也写出 `--output`，格式与批量结果兼容。
3. 输出中包含完整概率分布和 PTM 解析摘要。

### P1-3: 预处理后空数据集没有在入口层 fail-fast

证据:

- `clean_sequences` 可在无效序列时删除全部样本并返回空 DataFrame: `src/data/preprocess.py:130-137`。
- Lightning DataLoader 在空训练集上最终抛出底层 `ValueError: num_samples should be a positive integer`: `src/data/lightning_datamodule.py:165-172`。
- 实测当 `max_sequence_length=50` 而样本序列更长时，500 条样本全部被删除，错误直到 DataLoader 初始化才暴露。

影响:

配置错误或数据不匹配会在较晚阶段以底层错误暴露，排查成本高；自动化训练任务难以给出可操作错误。

解决策略:

1. `preprocess_pipeline` 后立即校验 train/val/test 非空，尤其是 train 非空。
2. 对“过滤后样本数为 0”抛出带上下文的异常:
   - 原始样本数。
   - 被过滤原因统计。
   - 当前 `max_sequence_length`。
3. CLI 捕获该异常并返回明确错误码。

### P2-1: 测试不足以证明 E2E 合约

证据:

- `tests/integration/test_train_lightning_script.py` 只验证 `--help` 能运行。
- `tests/unit/test_scripts.py` 对 `predict.py` 主要覆盖 parser、py_compile、batch helper，不覆盖真实 checkpoint。
- `tests/integration/test_lightning_pipeline.py` 覆盖 Lightning 组件训练，但未验证产物进入 CLI/API 推理。

影响:

当前测试绿灯不能证明用户可完成真实训练和推理闭环。

解决策略:

1. 新增 `tests/e2e/test_train_predict_cli.py`:
   - 原生训练 -> 推理。
   - Lightning 训练 -> 导出裸权重 -> 推理。
   - 断言标签映射一致。
2. 新增 `tests/e2e/test_api_initialize_checkpoint.py`:
   - 正确 checkpoint 成功。
   - 格式不匹配 checkpoint 失败。
   - Lightning checkpoint 转换成功。
3. 把 E2E 测试配置为小模型、小数据、CPU、1 epoch，确保 CI 可运行。

## 4. 推荐修复路线

### 阶段 1: 先修正语义安全问题

目标: 防止“能跑但结果错”。

任务:

1. 训练保存 `data.cell_states` 与 `label_to_idx`。
2. 推理缺少标签映射时 fail-fast。
3. API 初始化默认 strict 加载，权重不完整时失败。
4. 增加标签映射一致性测试和 API 错误 checkpoint 测试。

验收:

```bash
pytest -q tests/unit/test_training_inference_consistency.py tests/unit/test_scripts.py
pytest -q tests/e2e/test_train_predict_cli.py tests/e2e/test_api_initialize_checkpoint.py
```

### 阶段 2: 统一 checkpoint 合约

目标: 原生训练、Lightning 训练、CLI 推理、API 推理使用同一加载协议。

任务:

1. 实现统一 checkpoint 工具:
   - `save_inference_artifact(model, config, label_mapping, path)`
   - `load_inference_artifact(path, config_path=None)`
   - `extract_model_state_dict(checkpoint)`
2. `train.py` 和 `train_lightning.py` 都导出推理 artifact。
3. `predict.py`、API auto-init、API `/initialize` 只通过该工具加载。

验收:

```bash
python scripts/train.py --config configs/default.yaml --data data/raw/sample_data.csv --output /tmp/native --epochs 1 --batch_size 8
python scripts/predict.py --model /tmp/native/models/best_model.pt --sequence ACDEFGHIKLMNPQRSTVWY --device cpu

python scripts/train_lightning.py --config <small_config> --data data/raw/sample_data.csv --max-epochs 1 --batch-size 16 --precision 32
python scripts/predict.py --model <lightning_exported_best_model.pt> --sequence ACDEFGHIKLMNPQRSTVWY --device cpu
```

### 阶段 3: 补齐 PTM-aware 推理输入

目标: CLI/API/批量推理都真实消费 PTM 位点。

任务:

1. `predict.py --sequence` 支持 `--ptm-sites`。
2. `predict.py --input` 解析 `ptm_sites` JSON 列。
3. 输出包含 PTM 解析数量和非法位点统计。
4. 增加同序列不同 PTM 输入的回归测试。

验收:

```bash
python scripts/predict.py \
  --model <artifact> \
  --sequence ACDEFGHIKLMNPQRSTVWY \
  --ptm-sites '[{"position":3,"type":"phosphorylation"}]' \
  --device cpu
```

### 阶段 4: 改善数据与配置错误可诊断性

目标: 训练入口对数据问题 fail-fast 并给出可操作信息。

任务:

1. 预处理统计过滤原因。
2. train/val/test 为空时抛出领域错误。
3. CLI 捕获并打印数据路径、配置项、过滤统计。

验收:

```bash
python scripts/train_lightning.py --config <max_sequence_length_too_small_config> --data data/raw/sample_data.csv
```

应直接报出“过滤后训练集为空，可能原因是 max_sequence_length 太小”，而不是 DataLoader 的 `num_samples=0`。

## 5. 当前是否满足 E2E 要求

不满足。

更准确地说:

1. 原生 `train.py -> predict.py` 有运行级闭环，但由于标签映射没有持久化，语义上不可靠。
2. Lightning `train_lightning.py -> predict.py` 不闭环，checkpoint 格式不兼容。
3. API 初始化存在误报成功风险，不能作为可靠生产推理入口。
4. PTM-aware 推理输入在 CLI 中不完整，批量路径直接丢弃 PTM 位点。
5. 测试缺少真实 E2E 合约，当前绿灯不能证明交付质量。

建议先处理 P0-1、P0-2、P0-3，再补 P1 推理输入和数据诊断。完成这些后，项目才可以宣称满足基础 E2E 训练与推理要求。

## 6. 审计限制

项目当前没有可用 `.codegraph/` 索引，CodeGraph MCP 返回 `CodeGraph not initialized in /home/scu/PTM2CellNet`。本次审计使用本地文件读取、命令行执行和测试结果作为证据。
