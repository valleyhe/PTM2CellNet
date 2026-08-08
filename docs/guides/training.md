# 训练指南

## 概述

本指南介绍如何训练、调优和评估PTM2CellNet模型。

## 训练流程

### 1. 准备数据

```bash
# 运行数据整合
python scripts/integrate_data_v2.py --max-seq-len 1000

# 或使用现有数据
ls data/processed/ptm_*.csv
```

### 2. 选择配置文件

| 配置文件 | 适用场景 |
|---------|---------|
| `configs/default.yaml` | Transformer基线 |
| `configs/cnn_gated_fusion_112k.yaml` | 轻量级CNN |
| `configs/mamba_small.yaml` | 长序列Mamba |
| `configs/production.yaml` | 生产环境 |

### 3. 运行训练

```bash
# 使用默认配置
python scripts/train.py

# 自定义参数
python scripts/train.py \
  --config configs/default.yaml \
  --data data/processed/ptm_integrated_human_labeled.csv \
  --epochs 50 \
  --batch-size 64
```

## 模型调优

### 关键超参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|---------|------|
| `hidden_dim` | 256 | 64-512 | 隐藏层维度 |
| `num_layers` | 6 | 2-12 | 编码器层数 |
| `num_heads` | 8 | 4-16 | 注意力头数 |
| `dropout` | 0.1 | 0.1-0.3 | Dropout率 |
| `learning_rate` | 1e-3 | 1e-4-1e-2 | 学习率 |
| `batch_size` | 32 | 16-128 | 批次大小 |

### 使用Focal Loss处理类别不平衡

```yaml
# configs/training/focal_loss.yaml
training:
  loss_type: "focal"
  focal_alpha: 0.75
  focal_gamma: 2.0
```

## 评估模型

```bash
# 运行评估
python scripts/evaluate.py \
  --model outputs/models/best_model.pt \
  --data data/processed/test.csv

# 生成可视化
python scripts/evaluate.py --model outputs/models/best_model.pt --plot
```

## 使用GPU训练

```bash
# 指定设备
python scripts/train.py --device cuda

# 混合精度训练 (需PyTorch 1.6+)
# 在配置文件中启用
training:
  mixed_precision: true
```

## 断点续训

```bash
# 从完整训练状态恢复（会恢复 model/optimizer/scheduler/scaler/RNG）
python scripts/train.py --resume outputs/models/checkpoint_best.pt
```

`checkpoint_best.pt` 和 `checkpoint_best_last.pt` 是完整训练状态；
`best_model.pt` 是从最佳 checkpoint 导出的裸 `state_dict`，供 `predict.py`/API
推理使用。旧版仅含权重的 checkpoint 仍可加载，但不等价于精确续训。
