# PTM2CellNet 深度技术分析文档（基于当前代码实现）

**文档版本：** v5.0  
**分析基线：** `/home/scu/PTM2CellNet` 当前工作区代码  
**更新日期：** 2026-03-11  
**分析范围：** `src/`, `scripts/`, `configs/`, `tests/`, `docs/`  
**目标：** 以“实际代码行为”为准，完整呈现系统结构、算法逻辑、工程状态与技术风险。

---

## 1. 项目技术快照

### 1.1 代码规模与模块分布

根据 `find src -name '*.py' | xargs wc -l` 统计：

- `src/` 总计约 **5373 行** Python 代码。
- 模块行数（近似）：
  - `src/data`: 1238 行
  - `src/models`: 1073 行
  - `src/training`: 1228 行
  - `src/evaluation`: 666 行
  - `src/utils`: 654 行
  - `src/api`: 511 行

结论：当前项目已形成完整的端到端链路，不再是单一实验脚本，而是具备训练、评估、服务化接口的中型工程代码库。

### 1.2 技术栈现状

- 深度学习：`torch`（主干） + `pytorch-lightning`（可选训练栈）
- 预训练生态：`transformers`（ESM-2 / ProtBERT / ProtT5）
- Mamba实现依赖：`einops`
- 数据处理：`pandas`, `numpy`, `scikit-learn`, `biopython`
- API服务：`fastapi`, `pydantic`
- 可视化：`matplotlib`, `seaborn`

### 1.3 当前架构核心结论

- 同时存在两套训练路径：
  - 自定义 `Trainer`（`src/training/trainers.py`）
  - Lightning 路径（`src/training/lightning_module.py` + `src/data/lightning_datamodule.py`）
- 模型层支持四类编码器族：
  - 传统：CNN / Transformer / LSTM
  - Mamba：`src/models/mamba_encoder.py`
  - 预训练：ESM-2 / ProtBERT / ProtT5
- 数据输入输出契约相对稳定：batch 核心字段围绕 `sequence / ptm_types / ptm_mask / label`。

---

## 2. 全局架构与端到端数据流

### 2.1 离线训练主路径（自定义 Trainer）

脚本：`scripts/train.py`

1. `Config.from_yaml` 加载配置并应用 CLI 覆盖。  
2. `DataLoader` 读取 CSV 或生成 sample data。  
3. `DataPreprocessor.preprocess_pipeline` 执行清洗、PTM规范化、去重、划分。  
4. `PTMDataModule`（非 Lightning）构造 DataLoader。  
5. `PTM2CellNet.from_config` 创建模型。  
6. `Trainer.compile` 绑定损失/优化器/调度器。  
7. `Trainer.fit` 训练并触发回调。  
8. `Evaluator.evaluate` 测试集评估。  
9. `plot_training_curves` 导出训练曲线。

### 2.2 Lightning 训练路径

脚本：`scripts/train_lightning.py`, `scripts/train_pretrained.py`

1. `src/data/lightning_datamodule.PTMDataModule` 在 `setup(stage)` 里构建数据集。  
2. `PTM2CellNetLightning` 封装模型与优化器调度。  
3. 通过 `pl.Trainer` 执行 fit/test，集成 callback、logger、混合精度、策略配置。

### 2.3 在线/离线推理路径

- 在线：`FastAPI` 路由 `POST /api/v1/predict`、`POST /api/v1/batch_predict`。  
- 离线：`scripts/predict.py`。

统一关键逻辑：

1. `clean_sequence + validate_sequence` 校验输入序列。  
2. PTM位点通过 `validate_ptm_site` 过滤并编码。  
3. 模型输出概率后映射为 cell state 标签。  
4. 返回 `confidence + probabilities + 耗时`。

---

## 3. 模块级深度分析

## 3.1 `src/data`：数据层

### 3.1.1 `loaders.py`

核心能力：

- `load_from_csv`: CSV -> DataFrame
- `load_from_fasta`: FASTA -> `Dict[id, sequence]`，并做 `clean_sequence + validate_sequence`
- `load_from_json`: JSON -> `List[Dict]`，对 root 类型和数组项类型做约束
- `load_sample_data`: 生成可训练样本（用于快速验证）
- `load_combined_data`: 多文件合并（merge/concat）

实现要点：

- FASTA 小写序列会被转大写后验证。
- JSON 若非对象或对象数组会抛 `TypeError`。
- 多表合并策略基于“公共列”自动决策，灵活但在复杂 schema 下存在误合并风险。

### 3.1.2 `preprocess.py`

预处理管线：

- `clean_sequences`
- `remove_duplicate_sequences`
- `normalize_ptm_labels`
- `split_dataset`

细节：

- 支持小样本回退策略：当样本不足时走简化划分，避免 `train_test_split` 直接崩溃。
- 比例和不为 1 时自动归一化。
- 分层切分失败会自动回退为非分层。

### 3.1.3 `features.py`

支持特征：

- One-hot 序列编码
- k-mer 频率向量
- 理化统计特征（疏水性、带电、组成）
- PTM位置矩阵 + PTM计数特征
- 多种特征拼接（`combine_features`）

逻辑特征：

- `sequence_encoding` 支持 `onehot / kmer / both`。
- PTM JSON 解析失败时安全回退为空集合。
- `extract_features_for_sample` 允许无 PTM 输入。

### 3.1.4 `datasets.py` 与 `lightning_datamodule.py`

- `PTMDataset` 输出训练批次字段：
  - `sequence: LongTensor[max_len]`
  - `sequence_length: LongTensor[1]`
  - `ptm_mask: FloatTensor[max_len]`
  - `ptm_types: LongTensor[max_len]`
  - `label: LongTensor[1]`
- 非 Lightning `PTMDataModule` 在 `__init__` 即构建 dataset。  
- Lightning `PTMDataModule` 在 `setup(stage)` 构建 dataset。

当前状态：两套 DataModule 并存，接口相似但生命周期不同，增加维护成本。

---

## 3.2 `src/models`：模型层

### 3.2.1 `encoders.py`

实现了统一基类 `SequenceEncoder`，并提供：

- `CNNEncoder`：Embedding + Conv1d
- `TransformerEncoder`：Embedding + PositionEmbedding + `nn.TransformerEncoder`
- `LSTMEncoder`：Embedding + LSTM + 线性投影回 embed_dim

### 3.2.2 `mamba_encoder.py`

关键组件：

- `SelectiveSSM`
  - 输入投影分支（SSM支路 + 门控支路）
  - depthwise causal Conv1d
  - 输入依赖参数 (`delta`, `B`, `C`) 生成
  - 显式状态递推（Python for 循环）
- `MambaBlock`
  - LayerNorm -> SelectiveSSM -> Dropout -> Residual
- `MambaEncoder`
  - Embedding -> 多层 MambaBlock -> Final LayerNorm
  - 可选 mask 将 padding 区域归零

实现性质：

- 逻辑完整，可训练可反传。
- 递推在 Python 层显式循环，长序列下吞吐不如 fused kernel 实现。

### 3.2.3 `ptm_modules.py`

- `PTMEmbedding`：PTM类型嵌入 + 位置嵌入
- `PTMAttention`：`MultiheadAttention(query=sequence, key/value=ptm)` + 残差归一化
- `PTMModule`：多层 PTM attention 堆叠

稳健性处理：

- 对“全mask样本”做防护，避免 attention key 全失效导致异常行为。

### 3.2.4 `predictors.py`

- `ClassificationPredictor`: MLP + Linear -> logits/probabilities/predictions
- `RegressionPredictor`: MLP + Linear -> predictions

### 3.2.5 `architectures.py`（系统总装）

`PTM2CellNet.forward` 主逻辑：

1. `seq_emb = encoder(sequence)`
2. 构造/补全 `ptm_types`, `ptm_mask`, `ptm_positions`
3. `fused = PTMModule(seq_emb, ptm_*)`
4. `pooled = mean(fused, dim=1)`
5. `predictor(pooled)`

架构优势：

- 编码器可替换，PTM融合路径统一。

当前已识别风险（详见第7章）：

- 预训练编码器路径存在“模型尺寸解析”和“维度对齐”隐患。

---

## 3.3 `src/training`：训练层

### 3.3.1 自定义 Trainer 栈

文件：`trainers.py`, `losses.py`, `optimizers.py`, `callbacks.py`

能力：

- 标准 train/val epoch 循环
- scheduler 分支处理（含 `ReduceLROnPlateau`）
- callback 扩展点（checkpoint / early stopping）
- `predict` 推理接口

损失函数：

- `FocalLoss`（支持 ignore_index）
- `DiceLoss`
- `MultiTaskLoss`

优化器工厂：

- `adam`, `adamw`, `sgd`
- scheduler: `cosine`, `plateau`, `step`

### 3.3.2 Lightning 栈

文件：`lightning_module.py`

- `training_step/validation_step/test_step` 内直接使用 `cross_entropy`
- 优化器支持 `adamw`, `lion`（可选依赖）, `adam`, `sgd`
- scheduler 支持 `cosine`, `plateau`, `step`

当前状态：

- Lightning 栈与自定义 Trainer 栈功能重叠，存在策略漂移风险（例如损失函数默认值、调度器细节并不完全一致）。

---

## 3.4 `src/evaluation`：评估层

### 3.4.1 `metrics.py`

封装分类/回归指标：

- 分类：accuracy, precision/recall/f1（macro/micro/weighted）, auc_roc, auc_pr, confusion_matrix, classification_report
- 回归：mae, mse, rmse, r2

### 3.4.2 `evaluators.py`

`Evaluator.evaluate`：

1. 遍历 dataloader 收集预测、概率、标签
2. classification/regression 分支计算指标
3. 可选返回 `predictions/targets/probabilities`

### 3.4.3 `visualization.py`

绘图函数覆盖：

- ROC、PR、混淆矩阵
- 训练曲线
- 特征重要性
- 注意力热图

工程细节：

- 统一自动创建父目录。
- 后端固定 `Agg`，适配无显示环境。

---

## 3.5 `src/utils`：基础设施层

- `config.py`: YAML 配置加载、点路径读写、保存
- `helpers.py`: 序列/PTM/label 校验与清洗
- `io.py`: pickle/json/csv/npy/model I/O，含 `_SafeUnpickler`
- `logging.py`: logger 构建与时间戳日志文件名

亮点：

- `io._SafeUnpickler` 对 pickle 加载做白名单限制，降低反序列化风险。

---

## 3.6 `src/api`：服务层

- `schemas.py`: 请求/响应模型
- `routes.py`: 模型状态、预处理、预测、健康检查、模型信息
- `app.py`: FastAPI app factory + CORS + lifespan

接口：

- `POST /api/v1/predict`
- `POST /api/v1/batch_predict`
- `GET /api/v1/health`
- `GET /api/v1/model_info`

服务行为：

- 模型未初始化返回 503。
- 输入序列非法返回 400。
- 内部异常返回 500。

---

## 4. 核心算法逻辑与复杂度

| 算法/流程 | 代码位置 | 核心逻辑 | 时间复杂度（近似） | 主要瓶颈 |
|---|---|---|---|---|
| One-hot 编码 | `src/data/features.py` | 序列逐位映射 | O(L) | 长序列内存占用 |
| k-mer 统计 | `src/data/features.py` | 滑窗枚举 + base-k 索引 | O((L-k+1)\*k) | 维度指数增长 `|AA|^k` |
| PTM位置编码 | `src/data/features.py`, `src/data/datasets.py` | 位置-类型写入张量 | O(K) | 稀疏位点密集化 |
| Transformer 编码 | `src/models/encoders.py` | MHA + FFN | O(N^2\*D\*layers) | N^2 注意力 |
| LSTM 编码 | `src/models/encoders.py` | 逐步递推 | O(N\*D\*H\*layers) | 长序列吞吐 |
| SelectiveSSM | `src/models/mamba_encoder.py` | 输入依赖 SSM 参数 + 状态递推 | O(B\*L\*d_inner\*d_state) | Python层时序循环 |
| PTM跨注意力融合 | `src/models/ptm_modules.py` | seq query, ptm key/value | 约 O(B\*N^2\*D) | 长序列注意力 |
| 训练循环 | `src/training/trainers.py` | forward/backward/step | O(E\*Batches\*F/B) | 反向传播计算图 |
| 分类评估 | `src/evaluation/evaluators.py` | 汇总预测后计算多指标 | O(N)~O(N\*C) | 多分类 AUC/AP |

---

## 5. 配置体系与脚本编排

### 5.1 配置矩阵

- 通用默认：`configs/default.yaml`
- Lightning：`configs/lightning.yaml`
- Mamba：`configs/mamba_small.yaml`, `configs/mamba.yaml`, `configs/mamba_large.yaml`
- 预训练：`configs/pretrained/*.yaml`

### 5.2 脚本角色

- `scripts/prepare_data.py`: 生成 sample data + 预处理落盘
- `scripts/train.py`: 自定义 Trainer 路径
- `scripts/train_lightning.py`: Lightning 标准训练
- `scripts/train_pretrained.py`: 预训练模型训练入口
- `scripts/evaluate.py`: 评估与可视化
- `scripts/predict.py`: 单样本/批量离线推理

现状结论：

- 编排入口完整，但不同入口在默认行为上并非完全一致（例如标签来源、损失函数、参数覆盖方式）。

---

## 6. 测试与工程状态（基于当前环境）

### 6.1 测试组织

- 单元测试：`tests/unit/`
- 集成测试：`tests/integration/`
- 文档完整性测试：`tests/unit/test_analysis_doc_integrity.py`

### 6.2 当前环境下的可执行状态

执行 `pytest --collect-only -q` 结果：

- 已发现并收集 86 个测试用例节点。
- collection 阶段出现 7 个错误，主要来自环境依赖不完整/版本不兼容：
  - `einops` 缺失（影响 Mamba 相关模块导入）
  - `pytorch_lightning` 缺失（影响 Lightning 测试）
  - `torch.optim.lr_scheduler.LRScheduler` 在当前 torch 版本不可导入（兼容性问题）
  - 部分科学计算二进制包与 NumPy 2.x ABI 不匹配警告/错误（例如 `numexpr`, `bottleneck`）

结论：

- 测试体系较完整，但当前运行环境尚未满足“全链路测试可执行”前置条件。

---

## 7. 当前技术债务与风险清单

以下结论来自代码静态审计与行为推断，优先级按潜在影响排序。

### 7.1 高优先级

1. 序列编码 `0` 与 padding `0` 冲突风险  
- 位置：`src/data/datasets.py::_encode_sequence` + `src/models/encoders.py` embedding `padding_idx=0`。  
- 问题：氨基酸索引从 0 开始，`A` 与 padding 同号，语义混叠。

2. 预训练编码器模型尺寸解析不稳定  
- 位置：`src/models/architectures.py`。  
- 问题：`encoder_type.lower()` 后尺寸字符串变为 `8m/150m/650m`，与 `ESM2Encoder.MODEL_NAMES` 的 `8M/150M/650M` 键大小写不一致，易回退默认模型。

3. 预训练维度对齐隐患  
- 位置：`src/models/architectures.py`。  
- 问题：虽然 `self.embed_dim` 会更新为预训练 hidden dim，但 `PTMModule` 和 `predictor` 初始化时仍使用函数参数 `embed_dim`，可能与真实编码器输出维度不一致。

### 7.2 中优先级

4. 双训练栈并存导致行为漂移  
- 自定义 Trainer 与 Lightning 默认损失、调度策略、回调行为并不完全等价。

5. Mamba实现性能路径偏研究原型  
- `SelectiveSSM` 递推使用 Python for 循环，长序列吞吐可能受限。

6. 兼容性边界未收敛  
- `LRScheduler` 类型导入对 torch 版本敏感；依赖版本上界（尤其 NumPy / pandas 生态）未锁定。

### 7.3 低优先级

7. DataModule 重复实现  
- `src/data/datasets.py` 与 `src/data/lightning_datamodule.py` 重叠，认知成本高。

8. 脚本入口默认行为不统一  
- `predict.py` 初始化模型状态时未传入配置，可能出现 `max_sequence_length / ptm_types` 与训练配置不一致。

---

## 8. 建议的工程演进路线

### 8.1 第一阶段（稳定性修复，1-2周）

- 修正 token 编码与 padding 冲突（序列索引整体 +1，0 保留给 padding）。
- 修正预训练模型尺寸解析逻辑（大小写无关映射）。
- 统一预训练路径维度对齐（`PTMModule` 与 `predictor` 使用最终 `self.embed_dim`）。
- 优化优化器调度器类型兼容（兼容 torch 1.x/2.x）。

### 8.2 第二阶段（训练栈收敛，2-4周）

- 明确“主训练栈”并将另一套降级为兼容层，减少策略分叉。
- 整理 DataModule，统一 batch 契约和阶段生命周期。
- 为关键契约增加回归测试（预训练维度、token映射、API请求与训练输入一致性）。

### 8.3 第三阶段（性能与能力增强，持续）

- Mamba路径做 kernel/fused 优化或引入成熟实现。
- 加入结构信息（例如 AlphaFold 衍生特征）与更强 PTM-aware 表征。
- 扩展轨迹类评估指标，覆盖“状态转变”任务。

---

## 9. 结论

PTM2CellNet 当前代码库已具备完整的研究到工程闭环：

- 数据处理、模型组装、训练评估、API服务路径都已落地。
- 模块边界总体清晰，配置驱动能力完备。

同时，项目处于“功能完整但工程收敛未完成”的阶段：

- 已发现若干关键一致性风险（尤其预训练路径与token语义问题）。
- 测试体系较齐全，但环境依赖与版本兼容性仍需收敛。

若按第8章优先级推进，项目可从“可用研究工程”稳定过渡到“可持续迭代的生产级技术底座”。

