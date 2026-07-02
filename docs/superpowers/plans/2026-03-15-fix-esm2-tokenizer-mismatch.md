# ESM2 Tokenizer输入不匹配修复计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复ESM2编码器接收自定义氨基酸索引（1~20）而非ESM官方tokenizer输出的问题，使ESM2能够正确利用预训练权重。

**Architecture:**
- 在 `ESM2Encoder` 中集成官方tokenizer，并新增 `tokenize()` 方法对外暴露；
- 新增 `ESMTokenizedDataset` 类，在 `__getitem__` 时调用ESM tokenizer做序列编码，PTM mask/types 对齐到tokenized序列的偏移（跳过 `<cls>` token）；
- `PTMDataModule`（Lightning版）接受可选的 `tokenizer` 参数，当提供时使用新Dataset类；
- `train_pretrained.py` 在构建数据模块前从模型中取出tokenizer并传入。

**Tech Stack:** Python 3.10, PyTorch 2.x, HuggingFace transformers, PyTorch Lightning 2.x, pytest

---

## 问题背景

### 根本原因

`PTMDataset._encode_sequence()` 将氨基酸转为自定义索引（1~20，0为padding），但 `ESM2Encoder` 底层调用的是 HuggingFace 的 `AutoModel`，其 embedding 层期待 ESM tokenizer 的 token ID（vocab_size≈33，含 `<cls>=0, <pad>=1, <eos>=2` 等特殊 token）。

两套编码完全不兼容，导致ESM2预训练权重无法有效被利用。

### 修复策略

| 文件 | 修改内容 |
|------|---------|
| `src/models/pretrained_encoders.py` | `ESM2Encoder.__init__` 中加载并保存 `self.tokenizer`；新增 `tokenize(sequences: List[str]) -> dict` 方法 |
| `src/data/datasets.py` | 新增 `ESMTokenizedDataset` 类，使用ESM tokenizer编码序列，PTM mask对齐到正确偏移 |
| `src/data/lightning_datamodule.py` | `PTMDataModule.__init__` 增加可选 `tokenizer` 参数；`setup()` 中根据tokenizer选择Dataset类 |
| `scripts/train_pretrained.py` | 在创建 `PTMDataModule` 前，从 `model.encoder` 取出 `tokenizer` 传入 |

---

## 文件结构

```
src/
├── models/
│   └── pretrained_encoders.py   # 修改：添加tokenizer加载和tokenize()方法
├── data/
│   ├── datasets.py              # 修改：新增ESMTokenizedDataset类
│   └── lightning_datamodule.py  # 修改：接受tokenizer参数，按需切换Dataset
scripts/
└── train_pretrained.py          # 修改：传入tokenizer到PTMDataModule
tests/
└── unit/
    ├── test_esm_tokenizer.py    # 新建：ESM2Encoder.tokenize()单元测试
    ├── test_esm_dataset.py      # 新建：ESMTokenizedDataset单元测试
    └── test_datamodule_esm.py   # 新建：PTMDataModule(tokenizer=...)集成测试
```

---

## Chunk 1: ESM2Encoder 添加 tokenizer 支持

### Task 1: 为 ESM2Encoder 添加 tokenizer 加载与 tokenize 方法

**涉及文件:**
- 修改: `src/models/pretrained_encoders.py`
- 新建测试: `tests/unit/test_esm_tokenizer.py`

#### 背景知识

ESM2 使用 HuggingFace `AutoTokenizer`，调用方式：
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
encoded = tokenizer(["MKTAYIAKQRQISFVKSHFSRQ"], return_tensors="pt",
                    padding=True, truncation=True, max_length=1024)
# encoded["input_ids"] shape: [1, seq_len+2]  含<cls>和<eos>
# encoded["attention_mask"] shape: [1, seq_len+2]
```

ESM tokenizer 会在序列首尾各添加1个特殊token（`<cls>` 和 `<eos>`），因此：
- 原始序列长度 L → tokenized 长��� L+2
- 氨基酸位置 `i`（0-based）对应 tokenized 位置 `i+1`（跳过 `<cls>`）

- [ ] **Step 1.1: 写失败测试**

新建 `tests/unit/test_esm_tokenizer.py`，由 Codex 生成，测试内容：
1. `ESM2Encoder` 初始化后 `self.tokenizer` 不为 None
2. `tokenizer.tokenize("ACDEF")` 返回包含氨基酸token的列表（不含特殊token）
3. `encoder.tokenize(["ACDEF"])` 返回 dict，包含 `input_ids`、`attention_mask`
4. `input_ids.shape[1] == len("ACDEF") + 2`（含首尾特殊token）
5. `encoder.forward(input_ids)` 输出 shape 为 `[1, len("ACDEF")+2, hidden_dim]`

运行: `HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/test_esm_tokenizer.py -v`
预期: **FAIL**（tokenizer属性不存在）

- [ ] **Step 1.2: 实现 tokenizer 加载（由 Codex 完成）**

Codex 任务：修改 `src/models/pretrained_encoders.py` 中的 `ESM2Encoder.__init__`：
- 使用 `AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)` 加载 tokenizer
- 将 tokenizer 保存为 `self.tokenizer`
- 新增实例方法 `tokenize(sequences: List[str], max_length: int = 1024) -> dict`：
  - 调用 `self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)`
  - 返回包含 `input_ids` 和 `attention_mask` 的字典

**约束:**
- 不修改 `PretrainedEncoder.forward()` 签名
- 不修改其他编码器类（CNNEncoder/LSTMEncoder等）
- `tokenize()` 方法只在 `ESM2Encoder` 上定义，基类不需要

- [ ] **Step 1.3: 运行测试验证通过**

运行: `HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/test_esm_tokenizer.py -v`
预期: **PASS** 全部用例

- [ ] **Step 1.4: Commit**
```bash
git add src/models/pretrained_encoders.py tests/unit/test_esm_tokenizer.py
git commit -m "feat: add tokenizer support to ESM2Encoder"
```

---

## Chunk 2: 新建 ESMTokenizedDataset

### Task 2: 实现 ESMTokenizedDataset，正确处理 tokenized 序列和 PTM 对齐

**涉及文件:**
- 修改: `src/data/datasets.py`（追加新类）
- 新建测试: `tests/unit/test_esm_dataset.py`

#### 背景知识

PTM mask 对齐规则：
- ESM tokenizer 输出的 `input_ids` 形状：`[B, L_tok]`，其中 `L_tok = seq_len + 2`
- `<cls>` 在索引 0，`<eos>` 在最后
- 氨基酸位置 `i`（0-based, 原始序列）→ tokenized 位置 `i + 1`
- PTM mask 和 PTM types 张量需与 `input_ids` 等长，且 PTM 位置需 `+1` 偏移

`PTMDataset._encode_ptm()` 中：
```python
# 原始：pos = site["position"] - 1  （0-based）
# ESM：tokenized_pos = pos + 1       （跳过<cls>）
```

- [ ] **Step 2.1: 写失败测试**

新建 `tests/unit/test_esm_dataset.py`，由 Codex 生成，测试内容：
1. 使用 mock tokenizer（或真实ESM tokenizer），构建 `ESMTokenizedDataset`
2. `dataset[0]` 包含键：`input_ids`, `attention_mask`, `ptm_mask`, `ptm_types`, `label`
3. `input_ids.shape[0] == attention_mask.shape[0] == ptm_mask.shape[0]`（等长）
4. 当序列 `"ACDEF"` 有 PTM 在 position=3（1-based）时，`ptm_mask[3] == 1.0`（0-based tokenized索引=3，对应原始i=2，tokenized=3=2+1）
5. 无 PTM 时 `ptm_mask.sum() == 0`

运行: `HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/test_esm_dataset.py -v`
预期: **FAIL**（`ESMTokenizedDataset` 不存在）

- [ ] **Step 2.2: 实现 ESMTokenizedDataset（由 Codex 完成）**

Codex 任务：在 `src/data/datasets.py` 末尾追加 `ESMTokenizedDataset` 类：

**接口设计：**
```python
class ESMTokenizedDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, config=None):
        ...
    def __len__(self): ...
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 返回键：input_ids, attention_mask, ptm_mask, ptm_types, label（可选）
```

**实现要点：**
- `_tokenize_sequence(sequence: str) -> dict`：调用 `self.tokenizer([sequence], ...)` 返回 `input_ids[0]`, `attention_mask[0]`
- `_encode_ptm_esm(ptm_sites_json, tokenized_length)`: PTM位置 `+1` 偏移，mask和types长度与 `tokenized_length` 一致
- `_encode_label(label)`: 与 `PTMDataset` 逻辑相同，从 `df["cell_state"].unique()` 构建映射
- 截断处理：`max_length` 由 tokenizer 的 `truncation=True` 处理，ptm_mask也需截断到相同长度
- padding处理：已由tokenizer的 `padding=True` 统一处理

**约束:**
- 不修改已有的 `PTMDataset` 类（向后兼容）
- 不引入新的第三方依赖

- [ ] **Step 2.3: 运行测试验证通过**

运行: `HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/test_esm_dataset.py -v`
预期: **PASS**

- [ ] **Step 2.4: Commit**
```bash
git add src/data/datasets.py tests/unit/test_esm_dataset.py
git commit -m "feat: add ESMTokenizedDataset with correct tokenizer alignment"
```

---

## Chunk 3: PTMDataModule 支持 tokenizer 参数

### Task 3: 修改 PTMDataModule 以接受 tokenizer，按需切换 Dataset 类

**涉及文件:**
- 修改: `src/data/lightning_datamodule.py`
- 新建测试: `tests/unit/test_datamodule_esm.py`

- [ ] **Step 3.1: 写失败测试**

新建 `tests/unit/test_datamodule_esm.py`，由 Codex 生成，测试内容：
1. 不传 tokenizer 时，`PTMDataModule` 使用 `PTMDataset`（原有行为不变）
2. 传入 tokenizer 时，`PTMDataModule` 使用 `ESMTokenizedDataset`
3. `datamodule.setup("fit")` 后，`train_dataloader()` 第一个 batch 包含 `input_ids` 键
4. `input_ids.dtype == torch.long`
5. 验证 batch 中 `input_ids`, `attention_mask`, `ptm_mask`, `ptm_types` 形状一致

运行: `HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/test_datamodule_esm.py -v`
预期: **FAIL**

- [ ] **Step 3.2: 修改 PTMDataModule（由 Codex 完成）**

Codex 任务：修改 `src/data/lightning_datamodule.py`：
- `__init__` 增加参数 `tokenizer: Optional[Any] = None`（`Any` 兼容 HuggingFace tokenizer）
- 保存 `self.tokenizer = tokenizer`
- `setup()` 中：
  ```python
  if self.tokenizer is not None:
      # 使用 ESMTokenizedDataset
      from .datasets import ESMTokenizedDataset
      self.train_dataset = ESMTokenizedDataset(...)
  else:
      # 保持原有 PTMDataset（向后兼容）
      self.train_dataset = PTMDataset(...)
  ```
- 同样处理 `val_dataset` 和 `test_dataset`

**约束:**
- 不传 tokenizer 时行为完全不变（向后兼容）
- 不修改 DataLoader 参数

- [ ] **Step 3.3: 运行测试验证通过**

运行: `HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/test_datamodule_esm.py -v`
预期: **PASS**

- [ ] **Step 3.4: Commit**
```bash
git add src/data/lightning_datamodule.py tests/unit/test_datamodule_esm.py
git commit -m "feat: PTMDataModule accepts optional tokenizer for ESM datasets"
```

---

## Chunk 4: train_pretrained.py 接入 tokenizer

### Task 4: 训练脚本中从模型取出 tokenizer 并传入数据模块

**涉及文件:**
- 修改: `scripts/train_pretrained.py`

- [ ] **Step 4.1: 修改训练脚本（由 Codex 完成）**

Codex 任务：修改 `scripts/train_pretrained.py` 的 `main()` 函数：

在 `model = PTM2CellNet.from_config(...)` 之后，`PTMDataModule(...)` 之前，添加：
```python
# 如果使用ESM编码器，从模型中取出tokenizer
tokenizer = None
if hasattr(model.encoder, "tokenizer"):
    tokenizer = model.encoder.tokenizer
    logger.info(f"使用ESM tokenizer: {type(tokenizer).__name__}")
```

然后将 `tokenizer` 传入 `PTMDataModule`：
```python
data_module = PTMDataModule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    config=config.to_dict(),
    tokenizer=tokenizer,  # 新增
)
```

**约束:**
- 不修改其他逻辑
- 非ESM模型（CNN/LSTM/Mamba等）tokenizer=None，完全��后兼容

- [ ] **Step 4.2: 语法检查**

运行: `python -m compileall -q scripts/train_pretrained.py`
预期: 无输出（无语法错误）

- [ ] **Step 4.3: Commit**
```bash
git add scripts/train_pretrained.py
git commit -m "feat: pass ESM tokenizer from model to PTMDataModule in train_pretrained.py"
```

---

## Chunk 5: 端到端集成验证

### Task 5: 全链路测试，确认 tokenizer 修复有效

- [ ] **Step 5.1: 运行全部单元测试**

```bash
HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/test_esm_tokenizer.py tests/unit/test_esm_dataset.py tests/unit/test_datamodule_esm.py -v
```
预期: 全部 PASS

- [ ] **Step 5.2: 运行现有测试套件，确认无回归**

```bash
HF_ENDPOINT=https://hf-mirror.com pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```
预期: 原有测试全部 PASS，无新增失败

- [ ] **Step 5.3: 小批量冒烟训练（100步）**

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/train_pretrained.py \
  --model esm2_8M \
  --data data/processed/pmads_combined.csv \
  --max-epochs 1 \
  --batch-size 4 \
  --precision 32 \
  2>&1 | tail -40
```
预期:
- 日志显示 `"使用ESM tokenizer: EsmTokenizer"`
- 无 `KeyError`, 无 shape mismatch 错误
- 至少完成1个 training step，输出 `train_loss`

- [ ] **Step 5.4: 验证 input_ids 范围正确**

在冒烟训练中添加临时调试代码（或单独脚本）确认：
- `input_ids.max() < tokenizer.vocab_size`（约33）
- `input_ids.min() >= 0`

---

## 关键约束汇总

| 约束 | 说明 |
|------|------|
| 向后兼容 | 不传 tokenizer 时，所有原有行为不变 |
| 不修改 PTMDataset | 避免影响非ESM模型的训练流程 |
| 不修改 PretrainedEncoder 基类 forward 签名 | 避免影响 ProtBERT/ProtT5 |
| PTM 位置偏移必须正确 | tokenized_pos = raw_pos_0based + 1 |
| 截断一致性 | ptm_mask长度必须与tokenizer输出等长 |

## 验收标准

- [ ] 所有新增测试通过
- [ ] 原有测试零回归
- [ ] 冒烟训练成功运行1个epoch
- [ ] `input_ids` 值范围在 ESM vocab 范围内（< 33）
- [ ] 训练loss正常下降（无NaN/Inf）
