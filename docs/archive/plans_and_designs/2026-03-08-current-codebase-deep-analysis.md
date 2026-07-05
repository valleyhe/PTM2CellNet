# PTM2CellNet 当前代码库深度技术分析（2026-03-11）

> 目标：基于当前仓库真实实现，给出可追溯的结构分析、算法逻辑说明、同类方案对比和参考资源清单。

---

## 1. 全局架构与端到端数据流

### 1.1 训练链路（自定义 Trainer）

入口：`scripts/train.py`

1. 配置加载：`src/utils/config.py` (`Config.from_yaml`)  
2. 数据加载：`src/data/loaders.py` (`DataLoader`)  
3. 数据预处理：`src/data/preprocess.py` (`DataPreprocessor`)  
4. 数据封装：`src/data/datasets.py` (`PTMDataset`, `PTMDataModule`)  
5. 模型构建：`src/models/architectures.py` (`PTM2CellNet`)  
6. 训练执行：`src/training/trainers.py` (`Trainer`)  
7. 评估与可视化：`src/evaluation/evaluators.py`, `src/evaluation/visualization.py`

### 1.2 训练链路（Lightning）

入口：`scripts/train_lightning.py`, `scripts/train_pretrained.py`

1. 数据模块：`src/data/lightning_datamodule.py` (`PTMDataModule`)  
2. 模型封装：`src/training/lightning_module.py` (`PTM2CellNetLightning`)  
3. Trainer：`pytorch_lightning.Trainer` + callback/logger/precision 策略

### 1.3 在线推理链路（API）

入口：`src/api/app.py` + `src/api/routes.py`

- `POST /api/v1/predict`：单样本预测  
- `POST /api/v1/batch_predict`：批量预测  
- `GET /api/v1/health`：健康检查  
- `GET /api/v1/model_info`：模型信息

请求经 `clean_sequence + validate_sequence + validate_ptm_site` 预处理后进入模型前向。

---

## 2. 模块级分析

### 2.1 `src/data`

- `loaders.py`：支持 CSV/FASTA/JSON/sample data；FASTA 自动清洗与校验。  
- `preprocess.py`：序列清洗、PTM标准化、去重、分层划分（失败回退非分层）。  
- `features.py`：one-hot/k-mer/理化/PTM 特征提取与拼接。  
- `datasets.py`：样本张量化与 DataLoader 构建（非 Lightning）。

关键现状：两套 DataModule（`datasets.py` 与 `lightning_datamodule.py`）并存。

### 2.2 `src/models`

- `encoders.py`：CNN / Transformer / LSTM 编码器。  
- `mamba_encoder.py`：SelectiveSSM + MambaBlock + MambaEncoder。  
- `ptm_modules.py`：PTMEmbedding + PTMAttention + PTMModule。  
- `predictors.py`：分类/回归头。  
- `architectures.py`：统一端到端前向。

关键现状：预训练路径存在尺寸解析与维度对齐风险（需专项修复）。

### 2.3 `src/training`

- `trainers.py`：自定义训练循环。  
- `losses.py`：Focal/Dice/MultiTask。  
- `optimizers.py`：优化器与调度器工厂。  
- `callbacks.py`：Checkpoint/EarlyStopping。  
- `lightning_module.py`：Lightning 训练封装。

关键现状：两套训练栈共存，行为一致性需进一步收敛。

### 2.4 `src/evaluation`

- `metrics.py`：分类与回归指标封装。  
- `evaluators.py`：批量评估与输出聚合。  
- `visualization.py`：ROC/PR/混淆矩阵/训练曲线/注意力热图等。

### 2.5 `src/utils`

- `config.py`：配置读写与点路径访问。  
- `io.py`：文件读写与安全 unpickler。  
- `helpers.py`：序列/PTM/标签校验。  
- `logging.py`：统一日志。

### 2.6 `src/api`

- `schemas.py`：请求响应模型。  
- `routes.py`：模型状态管理、输入预处理、推理路由。  
- `app.py`：FastAPI 应用创建与 CORS。

---

## 3. 关键算法总览表

| 算法 | 模块位置 | 核心逻辑 | 时间复杂度（近似） | 风险点 |
|---|---|---|---|---|
| One-hot 序列编码 | `src/data/features.py` | 氨基酸位点独热映射 | O(L) | 长序列内存 |
| k-mer 统计 | `src/data/features.py` | 滑窗计数归一化 | O((L-k+1)\*k) | `20^k` 维度膨胀 |
| PTM 位点编码 | `src/data/datasets.py` | mask + type 写入 | O(K) | 稀疏信息密集化 |
| Transformer 编码 | `src/models/encoders.py` | self-attention + FFN | O(N^2\*D\*layers) | N^2 注意力瓶颈 |
| LSTM 编码 | `src/models/encoders.py` | 序列递推 | O(N\*D\*H\*layers) | 长序列效率 |
| SelectiveSSM | `src/models/mamba_encoder.py` | 输入依赖参数 + 状态递推 | O(B\*L\*d_inner\*d_state) | Python 递推循环 |
| PTM 融合注意力 | `src/models/ptm_modules.py` | seq query, ptm key/value | 约 O(B\*N^2\*D) | 长序列开销 |
| 训练循环 | `src/training/trainers.py` | forward/backward/step | O(E\*Batches\*F/B) | 反向图开销 |
| 分类评估 | `src/evaluation/evaluators.py` | 聚合后统一指标计算 | O(N)~O(N\*C) | 多分类 AUC/AP 开销 |

---

## 4. 同类方案横向比较

### 4.1 与通用蛋白大模型

- ESM/ProtBERT 侧重“超大规模预训练语义表示”，迁移能力强。  
- PTM2CellNet 当前优势是“工程可定制与端到端可控”，更适合中等资源快速迭代。

### 4.2 与 PTM-aware 新架构

- PTM-Mamba、PTMGPT2 将 PTM 作为一等 token 或显式机制，长序列与 PTM上下文表达更强。  
- PTM2CellNet 已支持 Mamba 编码器与 PTM 融合模块，但仍需进一步收敛预训练路径与性能优化。

### 4.3 与高工程化训练框架

- Lightning/OpenFold 等生态在分布式、监控、可恢复训练上更成熟。  
- PTM2CellNet 当前自定义 Trainer 逻辑清晰，适合快速实验，但多栈共存增加了维护负担。

---

## 5. 当前技术状态结论

1. 代码结构完整，`src/data` 到 `src/api` 已形成闭环。  
2. 算法层面覆盖传统编码器、Mamba、PTM注意力融合、分类预测。  
3. 工程层面存在关键收敛任务：
- 序列 token 与 padding 语义分离
- 预训练模型尺寸解析与维度对齐
- torch/lightning/einops 依赖兼容性治理

---

## 6. 参考资源

### 6.1 官方文档与框架

- PyTorch: https://pytorch.org/docs/stable/index.html
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- Transformers: https://huggingface.co/docs/transformers/index
- FastAPI: https://fastapi.tiangolo.com/
- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- NumPy: https://numpy.org/doc/stable/
- pandas: https://pandas.pydata.org/docs/
- BioPython SeqIO: https://biopython.org/wiki/SeqIO

### 6.2 相关模型与研究

- ESM repo: https://github.com/facebookresearch/esm
- ProtTrans/ProtBERT: https://arxiv.org/abs/2007.06225
- PTM-Mamba (Nat Methods): https://www.nature.com/articles/s41592-025-02656-9
- PTMGPT2 (Nat Commun): https://www.nature.com/articles/s41467-024-51071-9
- DeepMVP (Nat Methods): https://www.nature.com/articles/s41592-025-02797-x
- OpenFold: https://github.com/aqlaboratory/openfold
- MusiteDeep: https://github.com/duolinwang/MusiteDeep
- PTM-Psi: https://github.com/pnnl/PTMPSI
- DeepCellState: https://github.com/umarov90/DeepCellState

