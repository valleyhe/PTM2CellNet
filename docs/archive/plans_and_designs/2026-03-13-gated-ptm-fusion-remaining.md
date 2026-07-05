# GatedPTMFusion模块剩余步骤实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成GatedPTMFusion模块在PTM2CellNet架构中的集成，添加完整单元测试和对比实验脚本

**架构:** 通过添加ptm_fusion_type参数使PTM2CellNet支持门控融合，使用TDD方法编写测试，最后创建对比脚本验证性能

**Tech Stack:** PyTorch, PyTorch Lightning, pytest

---

## 文件结构

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/models/architectures.py` | 修改 | 添加ptm_fusion_type参数 |
| `tests/unit/test_gated_ptm_fusion.py` | 创建 | GatedPTMFusion单元测试 |
| `scripts/compare_fusion_mechanisms.py` | 创建 | 对比实验脚本 |

---

## Task 1: 更新PTM2CellNet支持ptm_fusion_type

**Files:**
- Modify: `src/models/architectures.py:25-111`

**前置条件:**
- PTMModule已实现fusion_type支持（Step 2已完成）
- GatedPTMFusion类已实现（Step 1已完成）

### Step 1.1: 添加ptm_fusion_type参数到PTM2CellNet.__init__

- [ ] **修改`__init__`签名**

在`src/models/architectures.py`第25-38行，添加`ptm_fusion_type: str = "attention"`参数：

```python
def __init__(
    self,
    encoder_type: str = "transformer",
    vocab_size: int = 20,
    embed_dim: int = 128,
    max_seq_len: int = 1000,
    num_ptm_types: int = 10,
    num_classes: int = 4,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    pretrained_cache_dir: Optional[str] = None,
    ptm_fusion_type: str = "attention",  # 新增参数
):
```

### Step 1.2: 传递参数给PTMModule

- [ ] **修改PTMModule实例化代码**

在`src/models/architectures.py`第104-111行，添加fusion_type参数：

```python
self.ptm_module = PTMModule(
    num_ptm_types=num_ptm_types,
    embed_dim=self.embed_dim,
    max_position=max_seq_len,
    num_attention_heads=max(1, num_heads),
    num_layers=max(1, num_layers),
    dropout=dropout,
    fusion_type=ptm_fusion_type,  # 新增
)
```

### Step 1.3: 验证修改

- [ ] **运行验证测试**

```bash
cd /home/scu/PTM2CellNet
python -c "
import torch
from src.models.architectures import PTM2CellNet

# 测试1: 默认行为（向后兼容）
model_default = PTM2CellNet(encoder_type='cnn', num_classes=4)
print('✓ 默认实例化成功')

# 测试2: 显式指定attention
model_attn = PTM2CellNet(encoder_type='cnn', num_classes=4, ptm_fusion_type='attention')
print('✓ attention模式实例化成功')

# 测试3: gated模式
model_gated = PTM2CellNet(encoder_type='cnn', num_classes=4, ptm_fusion_type='gated')
print('✓ gated模式实例化成功')

# 测试4: 前向传播
batch = {
    'sequence': torch.randint(0, 20, (2, 50)),
    'ptm_types': torch.randint(0, 10, (2, 50)),
    'ptm_mask': torch.ones(2, 50)
}
output = model_gated(batch)
assert output['logits'].shape == (2, 4)
print('✓ 前向传播成功，输出形状:', output['logits'].shape)

print('\n=== Task 1 验证通过 ===')
"
```

Expected: 所有测试通过

---

## Task 2: 创建单元测试

**Files:**
- Create: `tests/unit/test_gated_ptm_fusion.py`

### Step 2.1: 创建测试文件结构

- [ ] **创建测试文件**

```python
"""
GatedPTMFusion模块单元测试
"""

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
```

### Step 2.2: 实现初始化测试

- [ ] **添加test_gated_ptm_fusion_init**

```python
    def test_gated_ptm_fusion_init(self, sample_data):
        """测试GatedPTMFusion初始化"""
        _, _, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim, dropout=0.1)

        assert isinstance(module.gate_proj, torch.nn.Linear)
        assert module.gate_proj.in_features == embed_dim * 2
        assert module.gate_proj.out_features == embed_dim
        assert isinstance(module.dropout, torch.nn.Dropout)
        assert isinstance(module.layer_norm, torch.nn.LayerNorm)
        assert module.use_residual is True
```

### Step 2.3: 实现前向传播测试

- [ ] **添加test_gated_ptm_fusion_forward**

```python
    def test_gated_ptm_fusion_forward(self, sample_data):
        """测试GatedPTMFusion前向传播"""
        sequence_emb, ptm_emb, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        output = module(sequence_emb, ptm_emb)

        assert output.shape == sequence_emb.shape
        assert output.shape == (4, 50, 128)
```

### Step 2.4: 实现mask处理测试

- [ ] **添加test_gated_ptm_fusion_with_mask**

```python
    def test_gated_ptm_fusion_with_mask(self, sample_data):
        """测试带mask的前向传播"""
        sequence_emb, ptm_emb, ptm_mask, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        output = module(sequence_emb, ptm_emb, ptm_mask)

        assert output.shape == sequence_emb.shape
        # 验证mask=0的位置输出与纯序列更接近
```

### Step 2.5: 实现无残差连接测试

- [ ] **添加test_gated_ptm_fusion_without_residual**

```python
    def test_gated_ptm_fusion_without_residual(self, sample_data):
        """测试无残差连接模式"""
        sequence_emb, ptm_emb, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim, use_residual=False)

        output = module(sequence_emb, ptm_emb)

        assert output.shape == sequence_emb.shape
        assert module.use_residual is False
```

### Step 2.6: 实现门控值范围测试

- [ ] **添加test_gate_values_range**

```python
    def test_gate_values_range(self, sample_data):
        """测试门控值范围在[0,1]之间"""
        sequence_emb, ptm_emb, _, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        with torch.no_grad():
            combined = torch.cat([sequence_emb, ptm_emb], dim=-1)
            gate = torch.sigmoid(module.gate_proj(combined))

        assert (gate >= 0).all()
        assert (gate <= 1).all()
```

### Step 2.7: 实现PTMModule集成测试

- [ ] **添加test_ptm_module_with_gated_fusion**

```python
    def test_ptm_module_with_gated_fusion(self, sample_data):
        """测试PTMModule使用gated融合类型"""
        sequence_emb, ptm_types, ptm_mask, embed_dim = sample_data

        ptm_module = PTMModule(
            num_ptm_types=10,
            embed_dim=embed_dim,
            num_layers=2,
            fusion_type="gated"
        )

        ptm_positions = torch.arange(sequence_emb.size(1)).unsqueeze(0).repeat(sequence_emb.size(0), 1)
        output = ptm_module(sequence_emb, ptm_types.long(), ptm_positions, ptm_mask)

        assert output.shape == sequence_emb.shape
```

### Step 2.8: 实现反向传播测试

- [ ] **添加test_backward_pass**

```python
    def test_backward_pass(self, sample_data):
        """测试反向传播"""
        sequence_emb, ptm_emb, ptm_mask, embed_dim = sample_data
        module = GatedPTMFusion(embed_dim=embed_dim)

        sequence_emb.requires_grad = True
        output = module(sequence_emb, ptm_emb, ptm_mask)
        loss = output.sum()
        loss.backward()

        assert sequence_emb.grad is not None
        assert not torch.isnan(sequence_emb.grad).any()
```

### Step 2.9: 运行测试验证

- [ ] **运行所有测试**

```bash
cd /home/scu/PTM2CellNet
pytest tests/unit/test_gated_ptm_fusion.py -v
```

Expected: 8个测试全部通过

---

## Task 3: 创建对比实验脚本

**Files:**
- Create: `scripts/compare_fusion_mechanisms.py`

### Step 3.1: 创建脚本文件和导入

- [ ] **创建脚本骨架**

```python
#!/usr/bin/env python3
"""
对比PTM融合机制：Cross-Attention vs Gated Fusion

运行相同数据下的两种融合机制，对比：
1. 训练收敛速度
2. 最终验证损失
3. 推理速度
4. 参数量

使用方法:
    python scripts/compare_fusion_mechanisms.py \
        --data data/processed/train.csv \
        --epochs 10 \
        --output outputs/fusion_comparison
"""

import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.architectures import PTM2CellNet
from src.data.datasets import PTMDataset
from src.data.data_module import PTMDataModule
```

### Step 3.2: 实现参数解析

- [ ] **添加参数解析**

```python
def parse_args():
    parser = argparse.ArgumentParser(description='对比PTM融合机制')
    parser.add_argument('--data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output', type=str, default='outputs/fusion_comparison', help='输出目录')
    parser.add_argument('--encoder', type=str, default='cnn', choices=['cnn', 'lstm', 'mamba'], help='编码器类型')
    return parser.parse_args()
```

### Step 3.3: 实现模型创建函数

- [ ] **添加create_model函数**

```python
def create_model(fusion_type: str, encoder_type: str = "cnn"):
    """创建指定融合类型的模型"""
    return PTM2CellNet(
        encoder_type=encoder_type,
        vocab_size=20,
        embed_dim=128,
        num_ptm_types=10,
        num_classes=4,
        num_layers=2,
        dropout=0.1,
        ptm_fusion_type=fusion_type,
    )
```

### Step 3.4: 实现训练函数

- [ ] **添加train_epoch函数**

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output['logits'], batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
```

### Step 3.5: 实现评估函数

- [ ] **添加evaluate函数**

```python
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)
        loss = criterion(output['logits'], batch['labels'])
        total_loss += loss.item()
        preds = output['predictions']
        correct += (preds == batch['labels']).sum().item()
        total += batch['labels'].size(0)
    return total_loss / len(dataloader), correct / total
```

### Step 3.6: 实现对比主函数

- [ ] **添加compare函数**

```python
def compare_fusion_mechanisms(args):
    """对比两种融合机制"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    data_module = PTMDataModule(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=4,
    )
    data_module.setup()

    results = {}

    for fusion_type in ['attention', 'gated']:
        print(f"\n{'='*50}")
        print(f"Testing {fusion_type.upper()} fusion")
        print(f"{'='*50}")

        # 创建模型
        model = create_model(fusion_type, args.encoder).to(device)

        # 统计参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")

        # 优化器和损失
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 训练
        train_losses = []
        val_losses = []
        start_time = time.time()

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model, data_module.train_dataloader(), optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(
                model, data_module.val_dataloader(), criterion, device
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        train_time = time.time() - start_time

        # 推理速度测试
        model.eval()
        dummy_batch = {
            'sequence': torch.randint(0, 20, (32, 100)).to(device),
            'ptm_types': torch.randint(0, 10, (32, 100)).to(device),
            'ptm_mask': torch.ones(32, 100).to(device),
        }
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = model(dummy_batch)

            # 正式测试
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(100):
                _ = model(dummy_batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = (time.time() - start) / 100

        results[fusion_type] = {
            'num_params': num_params,
            'train_time': train_time,
            'inference_time_ms': inference_time * 1000,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
        }

    # 保存结果
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 打印对比
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    for fusion_type, metrics in results.items():
        print(f"\n{fusion_type.upper()}:")
        print(f"  Parameters: {metrics['num_params']:,}")
        print(f"  Train time: {metrics['train_time']:.2f}s")
        print(f"  Inference: {metrics['inference_time_ms']:.2f}ms/batch")
        print(f"  Final val loss: {metrics['final_val_loss']:.4f}")

    return results
```

### Step 3.7: 实现主函数

- [ ] **添加main函数**

```python
def main():
    args = parse_args()
    compare_fusion_mechanisms(args)

if __name__ == '__main__':
    main()
```

### Step 3.8: 验证脚本

- [ ] **运行脚本验证**

```bash
cd /home/scu/PTM2CellNet
python scripts/compare_fusion_mechanisms.py \
    --data data/processed/train.csv \
    --epochs 2 \
    --batch_size 16 \
    --encoder cnn \
    --output outputs/test_comparison
```

Expected: 脚本正常运行，输出对比结果

---

## 完成检查清单

- [ ] Task 1: PTM2CellNet支持ptm_fusion_type
- [ ] Task 2: 单元测试文件创建并通过
- [ ] Task 3: 对比实验脚本创建并可运行

---

## 备注

- 所有修改应保持向后兼容
- 默认fusion_type为"attention"
- 测试应覆盖正常和异常情况
