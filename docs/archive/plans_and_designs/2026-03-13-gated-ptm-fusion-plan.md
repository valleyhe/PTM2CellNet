# GatedPTMFusion模块实施计划

**日期**: 2026-03-13
**任务**: 实现门控PTM融合模块(GatedPTMFusion)
**执行方式**: 由codex CLI生成代码，Claude负责规划、检查、测试和合并

---

## 1. 任务概述

### 1.1 目标
实现一个门控机制的PTM融合模块，替代/增强现有的Cross-Attention融合，提供更细粒度的序列-PTM交互控制。

### 1.2 设计原理
```
输入: sequence_emb [B, L, D], ptm_emb [B, L, D]
              ↓
    ┌─────────────────────┐
    │   门控机制计算       │
    │  gate = σ(W_g·[seq;ptm] + b_g)  [B, L, D]
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │   特征融合          │
    │  output = gate·seq + (1-gate)·ptm  [B, L, D]
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │   残差连接 + LayerNorm
    └─────────────────────┘
```

---

## 2. 实施步骤

### 步骤1: 创建GatedPTMFusion类

**文件**: `src/models/ptm_modules.py`

**要求**:
```python
class GatedPTMFusion(nn.Module):
    """
    门控PTM融合模块

    通过可学习的门控机制动态控制序列特征和PTM特征的融合比例，
    实现更细粒度的特征交互。

    Args:
        embed_dim: 特征维度
        dropout: Dropout概率
        use_residual: 是否使用残差连接
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        # 1. 门控投影层：将[seq; ptm]映射到门控值
        # 2. Dropout层
        # 3. LayerNorm层
        # 4. 保存use_residual标志

    def forward(
        self,
        sequence_emb: torch.Tensor,
        ptm_emb: torch.Tensor,
        ptm_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            sequence_emb: [B, L, D] 序列嵌入
            ptm_emb: [B, L, D] PTM嵌入
            ptm_mask: [B, L] PTM掩码 (可选)

        Returns:
            output: [B, L, D] 融合后的特征
        """
        # 实现步骤：
        # 1. 拼接sequence_emb和ptm_emb -> [B, L, 2D]
        # 2. 通过门控投影层计算gate -> [B, L, D]
        # 3. 应用sigmoid激活
        # 4. 处理ptm_mask（如果提供）：将无效位置的gate设为0
        # 5. 计算融合特征：gate * sequence_emb + (1 - gate) * ptm_emb
        # 6. 应用dropout
        # 7. 残差连接（如果启用）：output + sequence_emb
        # 8. LayerNorm归一化
        # 9. 返回结果
```

**验收标准**:
- [ ] 类定义符合上述接口
- [ ] 支持ptm_mask处理
- [ ] 支持残差连接开关
- [ ] 包含完整的docstring

---

### 步骤2: 更新PTMModule以支持GatedPTMFusion

**文件**: `src/models/ptm_modules.py`

**要求**:
修改`PTMModule`类，添加`fusion_type`参数，支持切换融合机制：

```python
class PTMModule(nn.Module):
    def __init__(
        self,
        num_ptm_types: int,
        embed_dim: int,
        max_position: int = 1000,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "attention",  # 新增: "attention" 或 "gated"
    ):
        # 根据fusion_type选择使用PTMAttention还是GatedPTMFusion
```

**验收标准**:
- [ ] PTMModule支持`fusion_type`参数
- [ ] 向后兼容（默认使用attention）
- [ ] fusion_type="gated"时正确实例化GatedPTMFusion

---

### 步骤3: 更新PTM2CellNet架构支持

**文件**: `src/models/architectures.py`

**要求**:
在`PTM2CellNet.__init__`中添加`ptm_fusion_type`参数，并传递给PTMModule。

**验收标准**:
- [ ] PTM2CellNet支持ptm_fusion_type参数
- [ ] 参数正确传递给PTMModule
- [ ] 向后兼容

---

### 步骤4: 创建单元测试

**文件**: `tests/unit/test_gated_ptm_fusion.py` (新建)

**测试用例要求**:

```python
import pytest
import torch
from src.models.ptm_modules import GatedPTMFusion, PTMModule

class TestGatedPTMFusion:
    """GatedPTMFusion单元测试"""

    @pytest.fixture
    def sample_data(self):
        """测试数据fixture"""
        batch_size, seq_len, embed_dim = 4, 50, 128
        sequence_emb = torch.randn(batch_size, seq_len, embed_dim)
        ptm_emb = torch.randn(batch_size, seq_len, embed_dim)
        ptm_mask = torch.randint(0, 2, (batch_size, seq_len)).float()
        return sequence_emb, ptm_emb, ptm_mask, embed_dim

    def test_gated_ptm_fusion_init(self, sample_data):
        """测试GatedPTMFusion初始化"""
        # 验证模块能正常初始化

    def test_gated_ptm_fusion_forward(self, sample_data):
        """测试GatedPTMFusion前向传播"""
        # 验证输出形状正确 [B, L, D]

    def test_gated_ptm_fusion_with_mask(self, sample_data):
        """测试带mask的前向传播"""
        # 验证mask能正确处理

    def test_gated_ptm_fusion_without_residual(self, sample_data):
        """测试无残差连接模式"""
        # 验证use_residual=False工作正常

    def test_gated_values_range(self, sample_data):
        """测试门控值范围在[0,1]之间"""
        # 验证sigmoid输出正确

    def test_ptm_module_with_gated_fusion(self, sample_data):
        """测试PTMModule使用gated融合类型"""
        # 验证PTMModule(fusion_type="gated")正常工作

    def test_backward_pass(self, sample_data):
        """测试反向传播"""
        # 验证梯度能正常回传
```

**验收标准**:
- [ ] 所有测试用例通过
- [ ] 代码覆盖率>90%

---

### 步骤5: 创建对比实验脚本

**文件**: `scripts/compare_fusion_mechanisms.py` (新建)

**功能要求**:
```python
"""
对比PTM融合机制：Cross-Attention vs Gated Fusion

运行相同数据下的两种融合机制，对比：
1. 训练收敛速度
2. 最终验证损失
3. 推理速度
4. 参数量
"""

# 需要实现：
# 1. 使用相同配置训练两个模型（仅fusion_type不同）
# 2. 记录训练指标
# 3. 生成对比报告
# 4. 可视化对比结果
```

**验收标准**:
- [ ] 脚本能正常运行
- [ ] 输出对比报告

---

## 3. 测试验证计划

### 3.1 单元测试执行
```bash
pytest tests/unit/test_gated_ptm_fusion.py -v
```

### 3.2 集成测试
```bash
pytest tests/unit/test_models.py::TestPTMModules -v
```

### 3.3 快速训练验证
```bash
python scripts/train.py --config configs/mamba_tiny_gpu.yaml \
    --data data/processed/train.csv \
    --epochs 5 \
    --ptm_fusion_type gated
```

---

## 4. 代码审查清单

Claude在检查codex生成的代码时，需要验证：

### 4.1 功能正确性
- [ ] 门控计算使用sigmoid激活
- [ ] mask处理正确（无效位置gate=0）
- [ ] 残差连接可选且正确
- [ ] LayerNorm应用正确

### 4.2 代码质量
- [ ] 符合项目代码风格
- [ ] 包含完整docstring
- [ ] 类型注解完整
- [ ] 无硬编码值

### 4.3 兼容性
- [ ] 向后兼容（不影响现有代码）
- [ ] 支持所有现有配置
- [ ] 导出/导入功能正常

---

## 5. 合并检查清单

最终合并前确认：

- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 与现有PTMAttention对比实验完成
- [ ] 文档已更新
- [ ] 代码审查通过

---

## 6. 执行顺序

```
步骤1 → 步骤2 → 步骤3 → 步骤4 → 步骤5
  ↓       ↓       ↓       ↓       ↓
  └───────┴───────┴───────┴───────┘
              ↓
        测试验证 → 代码审查 → 合并
```

---

## 7. 回滚计划

如果发现问题：
1. 保留现有PTMAttention代码不变
2. 通过配置切换回`fusion_type="attention"`
3. 修复问题后重新测试

---

*计划创建时间*: 2026-03-13
*预计执行时间*: 2-3小时
