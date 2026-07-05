# PTM-Mamba架构设计文档

**文档版本：** v1.0
**创建日期：** 2026-03-10
**目标范围：** Stage 3 - PTM-Mamba架构实现
**优先级：** 高（Stage 3最高优先级）

---

## 1. 设计目标

### 1.1 核心目标

实现基于Mamba（State Space Model）的蛋白质序列编码器，以替代现有的Transformer编码器，实现：

1. **线性复杂度**：O(L)而非O(L²)，支持更长的蛋白质序列
2. **长程依赖**：更好地捕捉蛋白质序列的长程相互作用
3. **系统集成**：作为新的encoder选项无缝集成到现有系统
4. **向后兼容**：不破坏现有功能，保持所有接口一致

### 1.2 非目标

- 不实现预训练的Mamba模型（从头训练）
- 不修改PTM融合机制（复用现有PTMAttention）
- 不替换现有编码器（并行存在）

---

## 2. 架构设计

### 2.1 整体架构

```
PTM2CellNet (现有架构)
├── SequenceEncoder (扩展)
│   ├── CNNEncoder (现有)
│   ├── TransformerEncoder (现有)
│   ├── LSTMEncoder (现有)
│   └── MambaEncoder (新增) ⭐
│       ├── Embedding Layer
│       ├── MambaBlock × 12
│       │   ├── Input Projection
│       │   ├── Conv1D
│       │   ├── SelectiveSSM (核心)
│       │   ├── Output Projection
│       │   └── Residual + Norm
│       └── Final Norm
├── PTMModule (保持不变)
│   ├── PTMEmbedding
│   └── PTMAttention (跨注意力融合)
└── Predictor (保持不变)
```

**设计决策：**
- MambaEncoder输出与现有编码器相同的格式：`[B, L, hidden_dim]`
- PTM信息通过现有的PTMAttention模块融合（方���B）
- 完全复用现有的PTMModule和Predictor

### 2.2 Mamba核心机制

Mamba基于选择性状态空间模型（Selective SSM）：

```
输入序列 x[t] ∈ R^D
  ↓
计算选择性参数（输入依赖）:
  Δ[t] = softplus(Linear_Δ(x[t]))     # 时间步长
  B[t] = Linear_B(x[t])                # 输入矩阵
  C[t] = Linear_C(x[t]))               # 输出矩阵
  ↓
状态空间更新:
  h[t] = A̅·h[t-1] + B̅[t]·x[t]        # 离散化后的状态更新
  y[t] = C[t]·h[t] + D·x[t]           # 输出计算
  ↓
输出 y[t] ∈ R^D
```

**关键特性：**
- **线性复杂度**：O(L·D·N)，其中L是序列长度，D是模型维度，N是状态维度。相比Transformer的O(L²·D)，在长序列上有显著优势
- **选择性机制**：Δ, B, C根据输入动态调整，实现内容感知
- **长程依赖**：通过状态空间保持长距离信息

**复杂度对比：**
- Transformer: O(L²·D) - 序列长度的平方
- Mamba: O(L·D·N) - 线性于序列长度（N通常远小于L）
- 当L=1000, D=768, N=16时：Mamba约为Transformer的1/60计算量

---

## 3. 详细设计

### 3.1 MambaEncoder配置

**推荐配置（中等规模）：**

```yaml
model:
  encoder_type: "mamba"
  encoder_config:
    vocab_size: 20              # 20种氨基酸
    hidden_dim: 768             # 隐藏层维度
    num_layers: 12              # Mamba块数量
    state_dim: 16               # SSM状态维度
    conv_dim: 4                 # 1D卷积核大小
    expand_factor: 2            # 内部扩展因子
    dropout: 0.1                # Dropout率
```

**参数规模估算（详细分解）：**

```
配置：hidden_dim=768, state_dim=16, expand_factor=2

每个MambaBlock参数分解：
  d_inner = 768 × 2 = 1536

  1. Input Projection:
     - Linear(768, 1536×2): 768 × 3072 = 2,359,296

  2. Conv1D:
     - Conv1d(1536, 1536, kernel=4, groups=1536): 1536 × 4 = 6,144

  3. SelectiveSSM:
     - A parameter: 1536 × 16 = 24,576
     - D parameter: 1536
     - delta_proj: 1536 × 1536 = 2,359,296
     - B_proj: 1536 × 16 = 24,576
     - C_proj: 1536 × 16 = 24,576
     - 小计: 2,434,560

  4. Output Projection:
     - Linear(1536, 768): 1536 × 768 = 1,179,648

  5. LayerNorm:
     - 768 × 2 = 1,536

  每层总计: 2,359,296 + 6,144 + 2,434,560 + 1,179,648 + 1,536 ≈ 5,981,184 ≈ 6M

总参数量：
- Embedding: 20 × 768 = 15,360
- 12 × MambaBlock: 12 × 6M = 72M
- Final LayerNorm: 768 × 2 = 1,536
- 编码器总计: ≈ 72M

完整模型（含PTMModule + Predictor）：
- MambaEncoder: 72M
- PTMModule: ~0.3M
- Predictor: ~0.3M
- 总计: ≈ 72.6M
```

**注意**：实际参数量约72M，比初步估算的30M更大。这是因为expand_factor=2使得内部维度翻倍。

### 3.2 核心模块实现

**SelectiveSSM（状态空间模型核心）：**

```python
class SelectiveSSM(nn.Module):
    """选择性状态空间模型"""

    def __init__(
        self,
        d_model: int,      # 模型维度
        d_state: int = 16, # 状态维度
        d_conv: int = 4,   # 卷积维度
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # SSM参数（使用HiPPO初始化）
        # A矩阵：使用HiPPO-LegS初始化，确保长程记忆能力
        self.A = nn.Parameter(self._init_A(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def _init_A(self, d_model: int, d_state: int) -> torch.Tensor:
        """
        HiPPO-LegS初始化A矩阵

        A矩阵应该是负实数，确保状态空间稳定性
        使用HiPPO（High-order Polynomial Projection Operators）初始化
        可以更好地保持长程依赖

        简化版本：A[i,j] = -(2i+1) if i >= j else 0
        """
        A = torch.zeros(d_model, d_state)
        for i in range(d_model):
            for j in range(d_state):
                if i >= j:
                    A[i, j] = -(2 * j + 1)
        return A

        # 选择性投影
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

        # 1D卷积
        self.conv1d = nn.Conv1d(
            d_model, d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入序列
        Returns:
            y: [B, L, D] 输出序列
        """
        B, L, D = x.shape

        # 1. 1D卷积（因果卷积）
        x_conv = rearrange(x, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]  # 截断到原始长度
        x_conv = rearrange(x_conv, 'b d l -> b l d')

        # 2. 计算选择性参数（输入依赖）
        delta = F.softplus(self.delta_proj(x_conv))  # [B, L, D]
        B_param = self.B_proj(x_conv)  # [B, L, N]
        C_param = self.C_proj(x_conv)  # [B, L, N]

        # 3. 离散化SSM参数
        # 使用零阶保持（Zero-Order Hold）离散化
        # A̅ = exp(Δ·A), B̅ = (A̅ - I) / A · B
        A_bar = torch.exp(einsum(delta, self.A, 'b l d, d n -> b l d n'))
        B_bar = einsum(delta, B_param, 'b l d, b l n -> b l d n')

        # 4. 递推计算状态空间
        # h[t] = A̅·h[t-1] + B̅·x[t]
        # y[t] = C·h[t] + D·x[t]
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv[:, t:t+1, :]
            y_t = einsum(h, C_param[:, t], 'b d n, b n -> b d')
            y_t = y_t + self.D * x_conv[:, t]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, L, D]
        return y
```

**关键实现细节：**

1. **因果卷积**：使用padding确保只看到过去的信息
2. **Softplus激活**：确保Δ > 0，保持数值稳定性
3. **零阶保持离散化**：将连续SSM转换为离散递推形式
4. **递推计算**：逐时间步更新状态，保持O(L)复杂度

**数值稳定性技巧：**
- 使用softplus而非exp计算Δ，避免数值爆炸
- A矩阵使用特殊初始化（见下文）
- 可选：使用torch.float32而非float16进行SSM计算
```

**MambaBlock（Mamba层）：**

```python
class MambaBlock(nn.Module):
    """Mamba块"""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.ssm = SelectiveSSM(d_inner, d_state, d_conv)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        residual = x
        x = self.norm(x)

        # 分支投影
        x_proj = self.in_proj(x)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)

        # SSM处理
        x_ssm = self.ssm(x_ssm)

        # 门控
        x = x_ssm * F.silu(x_gate)

        # 输出投影 + 残差
        x = self.out_proj(x)
        return x + residual
```

**MambaEncoder（完整编码器）：**

```python
class MambaEncoder(nn.Module):
    """Mamba编码器"""

    def __init__(
        self,
        vocab_size: int = 20,
        hidden_dim: int = 768,
        num_layers: int = 12,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Mamba层堆叠
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                d_state=state_dim,
                d_conv=conv_dim,
                expand=expand_factor,
            )
            for _ in range(num_layers)
        ])

        # 最终归一化
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, L] token IDs
        mask: Optional[torch.Tensor] = None,  # [B, L]
    ) -> torch.Tensor:  # [B, L, hidden_dim]
        """
        前向传播

        Args:
            x: 输入token IDs [B, L]
            mask: 可选的padding mask [B, L]

        Returns:
            序列表示 [B, L, hidden_dim]
        """
        # 嵌入
        x = self.embedding(x)
        x = self.dropout(x)

        # Mamba层
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.norm(x)

        # 应用mask（处理padding）
        # mask: [B, L]，1表示有效token，0表示padding
        if mask is not None:
            # 将mask扩展到hidden_dim维度
            # [B, L] -> [B, L, 1] -> [B, L, hidden_dim]
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x * mask_expanded
            # 注意：padding位置的输出会被置零，不影响后续计算

        return x
```

**Mask机制说明：**
- mask在最后应用，确保padding位置的输出为0
- 这样在后续的pooling操作（如mean pooling）中，padding不会影响结果
- 如果需要更精细的mask控制，可以在每个MambaBlock内部应用

### 3.3 系统集成

**修改architectures.py：**

```python
from src.models.encoders import CNNEncoder, TransformerEncoder, LSTMEncoder
from src.models.mamba_encoder import MambaEncoder  # 新增

ENCODER_REGISTRY = {
    "cnn": CNNEncoder,
    "transformer": TransformerEncoder,
    "lstm": LSTMEncoder,
    "mamba": MambaEncoder,  # 新增
}

def create_encoder(encoder_type: str, config: dict):
    """创建编码器"""
    encoder_cls = ENCODER_REGISTRY.get(encoder_type)
    if encoder_cls is None:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return encoder_cls(**config)
```

**PTM2CellNet无需修改**，因为MambaEncoder实现了相同的接口。

---

## 4. 配置系统

### 4.1 配置文件

**configs/mamba.yaml（新增）：**

```yaml
# PTM-Mamba配置文件
model:
  encoder_type: "mamba"
  encoder_config:
    vocab_size: 20
    hidden_dim: 768
    num_layers: 12
    state_dim: 16
    conv_dim: 4
    expand_factor: 2
    dropout: 0.1

  ptm_module:
    num_ptm_types: 5
    ptm_embed_dim: 768
    num_attention_layers: 2
    num_heads: 8
    dropout: 0.1

  predictor:
    num_classes: 4
    hidden_dims: [768, 384]
    dropout: 0.1

data:
  max_seq_len: 1000          # 最大序列长度
  # 说明：
  # - 1000是蛋白质序列的典型长度（覆盖90%+的蛋白质）
  # - Mamba可以处理更长序列（2048+），但需要更多内存
  # - 超过max_seq_len的序列会被截断
  # - 短于max_seq_len的序列会被padding（用0填充）
  batch_size: 32
  num_workers: 4

training:
  max_epochs: 100
  learning_rate: 1e-4
  optimizer: "adamw"
  weight_decay: 0.01
  scheduler: "cosine"
  warmup_epochs: 5

  # Mamba特定优化
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2
  precision: "bf16-mixed"  # 混合精度训练（推荐）
  # 精度选择说明：
  # - "bf16-mixed": 推荐，平衡性能和稳定性
  # - "32-true": 如果遇到数值不稳定，使用全精度
  # - "16-mixed": 不推荐，可能导致SSM计算不稳定

  # Lightning配置
  accelerator: "gpu"
  devices: 1
  strategy: "auto"

callbacks:
  model_checkpoint:
    monitor: "val_loss"
    mode: "min"
    save_top_k: 3

  early_stopping:
    monitor: "val_loss"
    patience: 10
    mode: "min"
```

### 4.2 使用方式

**训练Mamba模型：**

```bash
# 使用Lightning训练脚本
python scripts/train_lightning.py \
    --config configs/mamba.yaml \
    --data data/raw/sample_data.csv \
    --output outputs/mamba_experiment

# 或使用专用脚本（可选）
python scripts/train_mamba.py \
    --config configs/mamba.yaml \
    --data data/raw/sample_data.csv
```

**推理：**

```bash
python scripts/predict.py \
    --model outputs/mamba_experiment/best_model.pt \
    --input data/test.csv \
    --output predictions.csv
```

---

## 5. 测试策略

### 5.1 单元测试

**tests/unit/test_mamba_encoder.py（新增）：**

```python
测试内容：
1. SelectiveSSM初始化和前向传播
2. MambaBlock的残差连接和归一化
3. MambaEncoder输出形状验证
4. 不同序列长度的处理能力
5. Mask机制正确性
6. 梯度反向传播
7. 与现有编码器接口一致性
```

### 5.2 集成测试

**tests/integration/test_mamba_pipeline.py（新增）：**

```python
测试内容：
1. 端到端训练流程（小数据集，10个epoch）
2. 与PTMModule的集成
3. Lightning训练兼容性
4. 模型保存和加载
5. 推理性能测试
6. 长序列处理（512, 1024, 2048）
```

### 5.3 对比实验

**基准对比：**

| 编码器 | 参数量 | 序列长度 | 训练时间 | 验证准确率 | 推理速度 |
|--------|--------|----------|----------|-----------|----------|
| Transformer | ~10M | 512 | 1x | baseline | 1x |
| ESM-2 150M | 150M | 1024 | 3x | +5% | 0.8x |
| Mamba (ours) | ~30M | 1024 | ? | ? | ? |

**成功标准：**
- ✅ 训练收敛，loss正常下降
- ✅ 验证准确率 ≥ Transformer baseline
- ✅ 长序列（>512）推理速度 > Transformer
- ✅ 内存占用合理（<20GB GPU）

---

## 6. 实施计划

### 6.1 开发阶段

**阶段1：核心实现（1-2周）**

任务：
1. 实现SelectiveSSM核心模块
   - SSM参数初始化
   - 离散化和状态更新
   - 数值稳定性处理
2. 实现MambaBlock
   - 输入/输出投影
   - 门控机制
   - 残差连接
3. 实现MambaEncoder
   - 层堆叠
   - 嵌入和归一化
4. 单元测试
   - 形状验证
   - 梯度检查
   - 接口一致性

**阶段2：系统集成（1周）**

任务：
1. 修改architectures.py注册Mamba
2. 创建配置文件（configs/mamba.yaml）
3. 集成测试
4. 文档更新（README, CLAUDE.md）

**阶段3：验证优化（1-2周）**

任务：
1. 小规模数据集训练验证
2. 与Transformer baseline对比
3. 超参数调优
   - 学习率
   - 层数和维度
   - Dropout率
4. 性能优化
   - 内存优化
   - 计算优化

### 6.2 里程碑

- **M1（2周）**：核心模块实现完成，单元测试通过
- **M2（3周）**：系统集成完成，集成测试通过
- **M3（4-5周）**：验证完成，性能达标

---

## 7. 风险和缓解

### 7.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| SSM数值不稳定 | 中 | 高 | 使用float32训练，添加数值裁剪 |
| 训练不收敛 | 中 | 高 | 参考Mamba论文的初始化策略 |
| 内存占用过大 | 低 | 中 | 使用梯度累积，减小batch size |
| 性能不如预期 | 中 | 中 | 调整超参数，或回退到Transformer |
| PyTorch实现慢 | 低 | 低 | 优化关键算子，使用torch.compile |

### 7.2 缓解策略

**数值稳定性：**
- 使用softplus而非exp计算Δ
- 对A矩阵使用特殊初始化（负实数）
- 添加epsilon防止除零

**训练稳定性：**
- 使用warmup学习率调度
- 梯度裁剪（clip_val=1.0）
- 较小的初始学习率（1e-4）

**性能优化：**
- 使用混合精度训练（bf16）
- 梯度���积减少内存
- torch.compile加速（PyTorch 2.0+）

---

### 7.3 数值稳定性详细说明

**关键数值范围：**
- Δ（时间步长）：限制在[0.001, 0.1]，使用`torch.clamp`
- 梯度范数：监控并记录，正常范围应在[0.1, 10]
- A矩阵：初始化为负值，范围约[-50, -1]
- 状态h：监控是否出现NaN/Inf

**精度选择策略：**
1. **首选bf16-mixed**：平衡性能和稳定性
2. **如果出现NaN**：切换到32-true全精度
3. **不推荐fp16-mixed**：可能导致SSM计算下溢

**调试检查清单：**
- [ ] 检查Δ值是否在合理范围
- [ ] 检查梯度范数是否正常
- [ ] 检查loss是否为NaN
- [ ] 检查状态h是否溢出
- [ ] 验证A矩阵初始化正确

---

## 8. 依赖和资源

### 8.1 新增依赖

```txt
# requirements.txt 新增
einops>=0.7.0              # 张量操作
causal-conv1d>=1.1.0       # 因果卷积（可选）
```

### 8.2 计算资源

**训练：**
- GPU：至少16GB显存（推荐24GB）
- 内存：32GB+
- 存储：10GB+（模型检查点）

**推理：**
- GPU：8GB显存或CPU
- 内存：16GB+

### 8.3 数据需求

- 训练集：至少1000个样本（推荐10000+）
- 验证集：200+样本
- 测试集：200+样本

---

## 9. 文档和交付物

### 9.1 代码文件

```
新增文件：
- src/models/mamba_encoder.py          # Mamba编码器实现
- configs/mamba.yaml                    # Mamba配置文件
- tests/unit/test_mamba_encoder.py     # 单元测试
- tests/integration/test_mamba_pipeline.py  # 集成测试
- scripts/train_mamba.py (可选)        # 专用训练脚本

修改文件：
- src/models/architectures.py          # 注册Mamba编码器
- requirements.txt                      # 添加依赖
- README.md                             # 更新文档
- CLAUDE.md                             # 更新开发指南
```

### 9.2 文档更新

- README.md：添加Mamba使用说明
- CLAUDE.md：更新架构说明
- 技术分析文档：更新Stage 3进度

### 9.3 测试报告

- 单元测试覆盖率报告
- 集成测试结果
- 性能对比报告
- 训练日志和可视化

---

## 10. 后续优化方向

### 10.1 短期优化（完成后）

1. **超参数搜索**：系统化调优层数、维度、学习率
2. **性能优化**：使用torch.compile或自定义CUDA kernel
3. **长序列测试**：验证2048+序列的处理能力

### 10.2 中期优化（3-6个月）

1. **门控融合**：实现方案D，在Mamba层内融合PTM信息
2. **预训练**：在大规模蛋白质数据集上预训练Mamba
3. **多任务学习**：扩展到多任务场景

### 10.3 长期优化（6-12个月）

1. **Mamba-2**：升级到最新的Mamba-2架构
2. **混合专家**：结合MoE提升模型容量
3. **论文发表**：整理实验结果发表论文

---

### 10.4 配置预设

**Small（轻量级）：**
```yaml
encoder_config:
  hidden_dim: 256
  num_layers: 6
  state_dim: 16
  # 参数量: ~10M
  # 适用: 快速实验、资源受限
```

**Medium（推荐）：**
```yaml
encoder_config:
  hidden_dim: 768
  num_layers: 12
  state_dim: 16
  # 参数量: ~72M
  # 适用: 生产环境、论文发表
```

**Large（高性能）：**
```yaml
encoder_config:
  hidden_dim: 1024
  num_layers: 24
  state_dim: 32
  # 参数量: ~300M
  # 适用: 大规模数据集、追求极致性能
```

---

### 10.5 常见问题（FAQ）

**Q1: Mamba相比Transformer有什么优势？**

A: 线性复杂度O(L·D·N) vs O(L²·D)，在长序列（>512）上速度和内存占用都更优。

**Q2: 训练时出现NaN怎么办？**

A:
1. 检查Δ值是否过大，添加裁剪
2. 切换到全精度训练（precision: "32-true"）
3. 降低学习率
4. 检查A矩阵初始化

**Q3: 如何验证实现正确性？**

A:
1. 与官方Mamba实现对齐（使用相同输入，比较输出）
2. 梯度检查（torch.autograd.gradcheck）
3. 小数据集过拟合测试

**Q4: 性能不如Transformer怎么办？**

A:
1. 检查超参数（学习率、层数、维度）
2. 尝试更长的训练时间
3. 检查数据预处理是否正确
4. 考虑使用预训练模型

**Q5: 如何处理超长序列（>2048）？**

A:
1. 使用滑动窗口切分
2. 增加梯度累积步数
3. 减小batch size
4. 考虑使用分布式训练

**Q6: 可以使用预训练的Mamba模型吗？**

A: 目前从头训练。未来可以考虑在大规模蛋白质数据集上预训练。

---

## 11. 参考文献

### 核心论文

1. **Mamba**: Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" arXiv 2023
   - arXiv: https://arxiv.org/abs/2312.00752
   - 官方实现: https://github.com/state-spaces/mamba

2. **S4**: Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces" ICLR 2022
   - arXiv: https://arxiv.org/abs/2111.00396
   - 官方实现: https://github.com/state-spaces/s4

3. **HiPPO**: Gu et al. "HiPPO: Recurrent Memory with Optimal Polynomial Projections" NeurIPS 2020
   - arXiv: https://arxiv.org/abs/2008.07669

### 蛋白质语言模型

4. **ESM-2**: Lin et al. "Language models of protein sequences at the scale of evolution" Science 2023
   - Paper: https://www.science.org/doi/10.1126/science.ade2574
   - 官方实现: https://github.com/facebookresearch/esm

5. **ProtBERT**: Elnaggar et al. "ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing" IEEE TPAMI 2021

### 实现参考

6. **mamba-minimal**: PyTorch原生实现
   - GitHub: https://github.com/johnma2006/mamba-minimal

7. **einops**: 张量操作库
   - Docs: https://einops.rocks/

### 项目文档

8. **PTM2CellNet技术分析文档**: docs/PTM2CellNet-深度技术分析文档.md
9. **Stage 3实施计划**: docs/superpowers/plans/2026-03-10-stage3-implementation-plan.md

---

**文档结束**

*本设计文档将作为PTM-Mamba架构实现的核心参考。*
