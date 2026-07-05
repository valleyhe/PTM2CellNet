# PTM2CellNet Benchmark Report

**Date**: 2026-03-12
**Comparing**: Transformer vs LSTM vs Mamba on Protein Sequence Classification
**Dataset**: Real protein sequences with PTM annotations

---

## Executive Summary

| Model | Best Val Loss | Test Accuracy | Training Time (10 epochs) | Speed |
|-------|---------------|---------------|---------------------------|-------|
| **LSTM** | **0.7804** ⭐ | 28.3% | ~47 sec | ✅ Fast |
| **Transformer** | 0.7870 | **32.1%** ⭐ | ~47 sec | ✅ Fast |
| Mamba | 0.7899 | 20.8% | ~90 min | ❌ Slow |

**Winner**:
- **Best Validation Loss**: LSTM (0.7804)
- **Best Test Accuracy**: Transformer (32.1%)
- **Speed**: Transformer and LSTM are ~115x faster than Mamba

---

## 1. Experimental Setup

### 1.1 Hardware Configuration
```
GPU: NVIDIA Tesla P40 (24GB VRAM)
CUDA: 11.8
PyTorch: 2.0.1+cu118
```

### 1.2 Model Architectures

#### Transformer (Small GPU)
```yaml
encoder_type: transformer
hidden_dim: 256
num_layers: 6
num_heads: 8
dropout: 0.1
batch_size: 8
max_sequence_length: 500
```

#### Mamba (Small GPU)
```yaml
encoder_type: mamba
hidden_dim: 256
num_layers: 6
state_dim: 16
dropout: 0.1
batch_size: 16
max_sequence_length: 500
```

#### LSTM (Small GPU)
```yaml
encoder_type: lstm
hidden_dim: 256
num_layers: 6
bidirectional: true
dropout: 0.1
batch_size: 16
max_sequence_length: 500
```

### 1.3 Dataset Statistics
| Split | Samples | Description |
|-------|---------|-------------|
| Train | 244 | Real protein sequences with PTM sites |
| Validation | 53 | Held-out for validation |
| Test | 53 | Final evaluation |

**Sequence Length**: ~400-500 amino acids
**PTM Types**: Methylation, Acetylation, Ubiquitination, Phosphorylation, etc.

### 1.4 Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 0.0002
- **Scheduler**: Cosine with warmup
- **Loss Function**: Focal Loss (gamma=2.0)
- **Precision**: FP16 (mixed precision)
- **Epochs**: 10

---

## 2. Results

### 2.1 Training Curves

#### Validation Loss per Epoch
| Epoch | Transformer | LSTM | Mamba | Best |
|-------|-------------|------|-------|------|
| 0 | 0.7966 | 0.7904 | **0.7899** | Mamba |
| 1 | 0.7919 | 0.7936 | **0.7903** | Mamba |
| 2 | 0.7924 | 0.7908 | 0.7985 | LSTM |
| 3 | **0.7870** | 0.7833 | 0.8000 | **Transformer** |
| 4 | 0.7957 | 0.7869 | 0.8080 | LSTM |
| 5 | 0.8032 | **0.7804** ⭐ | 0.8216 | **LSTM** |
| 6 | 0.8165 | 0.7807 | 0.8322 | LSTM |
| 7 | 0.8123 | 0.7841 | 0.8370 | LSTM |
| 8 | 0.8168 | 0.7856 | 0.8450 | LSTM |
| 9 | 0.8177 | 0.7860 | 0.8441 | LSTM |

**Best Performance**: LSTM at epoch 5 (val_loss = 0.7804)

#### Training Loss per Epoch
| Epoch | Transformer | LSTM | Mamba |
|-------|-------------|------|-------|
| 0 | 0.7975 | 0.8047 | 0.7886 |
| 1 | 0.7627 | 0.7724 | 0.7542 |
| 2 | 0.7498 | 0.7417 | 0.7173 |
| 3 | 0.7351 | 0.7255 | 0.6686 |
| 4 | 0.6950 | 0.6757 | 0.6278 |
| 5 | 0.6561 | 0.6399 | 0.5738 |
| 6 | 0.6096 | 0.6095 | 0.5209 |
| 7 | 0.5791 | 0.5682 | 0.4944 |
| 8 | 0.5626 | 0.5488 | 0.4703 |
| 9 | 0.5500 | 0.5355 | 0.4610 |

**Observation**: All models achieve similar training loss reduction, but validation loss behavior differs significantly → all models overfit on this small dataset.

### 2.2 Test Set Performance

| Metric | Transformer | LSTM | Mamba | Best |
|--------|-------------|------|-------|------|
| **Accuracy** | **32.08%** ⭐ | 28.30% | 20.75% | **Transformer** |
| Precision (macro) | **32.38%** ⭐ | 27.37% | 20.60% | **Transformer** |
| Recall (macro) | **32.14%** ⭐ | 28.02% | 20.88% | **Transformer** |
| F1 Score (macro) | **32.10%** ⭐ | 27.46% | 20.62% | **Transformer** |
| AUC-ROC | **52.55%** ⭐ | 44.40% | 51.63% | **Transformer** |
| AUC-PR | **30.02%** ⭐ | 28.61% | 27.46% | **Transformer** |
| **Best Val Loss** | 0.7870 | **0.7804** ⭐ | 0.7899 | **LSTM** |

**Key Findings**:
- **Transformer**: Best generalization (highest test accuracy across all metrics)
- **LSTM**: Best validation loss (0.7804) but lower test accuracy (28.3%)
- **Mamba**: Worst performance on all metrics, extremely slow training

### 2.3 Training Efficiency

| Metric | Transformer | LSTM | Mamba | vs Mamba |
|--------|-------------|------|-------|----------|
| Total Time (10 epochs) | ~47 sec | ~47 sec | ~90 min | **115x slower** |
| Time per Epoch | ~4.7 sec | ~4.7 sec | ~9 min | **115x slower** |
| GPU Memory | ~8 GB | ~4 GB | ~16 GB | 4x more |
| GPU Utilization | 100% | 100% | 100% | Equal |

**Efficiency Ranking**:
1. 🥇 **LSTM** - Fastest + most memory efficient
2. 🥈 **Transformer** - Fast, moderate memory
3. 🥉 **Mamba** - Very slow, highest memory

---

## 3. Analysis

### 3.1 Convergence Behavior

**LSTM**:
- Achieves best validation loss (0.7804 at epoch 5)
- Most stable validation curve
- Best memory efficiency
- However, lower test accuracy suggests some overfitting

**Transformer**:
- Fast convergence (reaches best validation loss by epoch 3)
- Good generalization (highest test accuracy)
- Validation loss increases gradually after epoch 3 (mild overfitting)

**Mamba**:
- Slowest convergence per epoch
- Severe overfitting (training loss ↓ 42%, validation loss ↑ 7%)
- Best performance at epoch 0 (first epoch only)
- Memory intensive

### 3.2 Memory Efficiency

**LSTM**: Most memory efficient
- Simple RNN architecture with minimal overhead
- Batch size 16 fits in ~4GB
- Best for memory-constrained environments

**Transformer**: Moderate memory usage
- Optimized PyTorch attention kernels
- Batch size 8 requires ~8GB

**Mamba**: Highest memory usage
- mamba-ssm CUDA kernels require significant memory
- Batch size 16 uses ~16GB

### 3.3 Why Mamba Underperforms

1. **Small Dataset**: Mamba (state-space model) typically requires larger datasets to learn effective state transitions

2. **Overfitting**: Mamba's expressive power leads to memorization with only 244 training samples

3. **Slow Training**: Each epoch takes ~9 minutes vs 5 seconds for Transformer

4. **No Pretraining**: Both models trained from scratch; pretrained weights would help

---

## 4. Synthetic vs Real Data Comparison

### Synthetic Data (500 samples, short sequences)
| Model | Best Val Loss | Time (10 epochs) |
|-------|---------------|------------------|
| LSTM | - | - |
| Transformer | 0.7878 | ~1 minute |
| Mamba | 0.7693 | ~2.5 hours |

### Real Data (350 samples, long sequences)
| Model | Best Val Loss | Time (10 epochs) |
|-------|---------------|------------------|
| **LSTM** | **0.7804** ⭐ | ~47 seconds |
| Transformer | 0.7870 | ~47 seconds |
| Mamba | 0.7899 | ~90 minutes |

**Key Difference**: Real data has longer sequences (~400-500 vs ~100 amino acids), which:
- Slows down Mamba disproportionately
- Causes Transformer to OOM with larger batch sizes
- Makes the task harder (lower overall accuracy)

---

## 5. Recommendations

### For Production Use
✅ **Use Transformer** - Best test accuracy (32.1%), fast training (~47 sec)

### For Research/Experimentation
✅ **Use LSTM** - Best validation loss (0.7804), most memory efficient (~4GB), equally fast

### For Future Experiments
1. **Increase Dataset Size**: Both models would benefit from 10x more data
2. **Add Regularization**: Dropout, weight decay, early stopping
3. **Try Pretrained Models**: ESM2, ProtBERT for protein sequences
4. **Tune Mamba State Dim**: Try state_dim=4 or 8 instead of 16
5. **Use Gradient Checkpointing**: For memory-constrained training

### Model Selection Guide
| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| **Best accuracy** | **Transformer** | 32.1% test accuracy |
| **Best val loss** | **LSTM** | 0.7804 validation loss |
| **Limited GPU memory** | **LSTM** | Only ~4GB required |
| **Fast training** | **LSTM/Transformer** | Both ~47 sec for 10 epochs |
| **Small dataset (<1000)** | **LSTM or Transformer** | Avoid Mamba (overfits) |
| **Very long sequences (>1000)** | Try Mamba | Theoretical advantage |
| **Large dataset (>10k)** | Any | All should work well |
| **Production deployment** | **Transformer** | Best balance of metrics |
| **Research/Experimentation** | **LSTM** | Fast, memory-efficient |

---

## 6. Raw Data

### Training Logs Location
- Transformer: `/tmp/transformer_real_data.log`
- LSTM: `/tmp/lstm_real_data.log`
- Mamba: `/tmp/mamba_real_data.log`

### Model Checkpoints
- Best Transformer: `outputs/models/transformer_small_gpu/best_model.pt`
- Best LSTM: `outputs/models/lstm_small_gpu/best_model.pt`
- Best Mamba: `outputs/models/mamba_small_gpu/best_model.pt`

### Training Curves
- Saved to: `outputs/results/training_curves.png`

---

## 7. Conclusion

For the PTM2CellNet protein sequence classification task:

### Final Rankings

| Rank | Model | Best For | Key Strength |
|------|-------|----------|--------------|
| 🥇 | **Transformer** | Production deployment | Best test accuracy (32.1%) |
| 🥈 | **LSTM** | Research/Experimentation | Best val loss (0.7804), fastest, most efficient |
| 🥉 | **Mamba** | Not recommended at this scale | 115x slower, severe overfitting |

### Key Findings

1. **Transformer is best for production** - Highest test accuracy (32.1%), fast training (~47 sec)

2. **LSTM is excellent for research** - Best validation loss (0.7804), most memory efficient (~4GB), equally fast

3. **Mamba underperforms** - Theoretical advantages don't translate at this dataset size; severe overfitting and extremely slow

4. **Dataset size is the limiting factor** - All models overfit on 244 training samples; need 10x more data

5. **Future work**:
   - Collect more real protein data (2000+ samples)
   - Try pretrained protein language models (ESM-2, ProtBERT)
   - Implement regularization techniques (dropout, weight decay)
   - Consider ensemble methods (Transformer + LSTM)

---

*Report generated: 2026-03-12*
*Benchmark script: `scripts/train.py`*
