# DAVF集成设计文档: PTM→信号通路效应预测

**日期**: 2026-05-03  
**方案**: 轻量级Pipeline注入 + 级联融合

---

## 1. 目标

将方向感知的DAVF(Direction-Aware Velocity Field)模型集成到PTM2CellNet中，实现：
- PTM修饰事件 → 映射为基因扰动(gene_id + direction)
- DAVF推理 → 预测下游基因表达变化(delta_expression)
- 级联融合 → delta_expression压缩为特征，与序列特征拼接后预测细胞状态

## 2. 架构概览

```
PTM输入(类型+位点+蛋白质)
       │
       ├──→ [PTMDirectionMapper] ──→ (gene_ids, directions, attention_mask)
       │                                        │
       │                                        ▼
       │                          控制表达 → [LatentDAVF/DAVF] → delta_expression
       │                                        │
       │                                        ▼
       │                              [DeltaProjection] → davf_features (128d)
       │                                        │
       ▼                                        │
序列 → Encoder → PTMModule → Pooling ──────→ [concat] → Predictor → 细胞状态
                                    (embed_dim)    (embed_dim + 128)
```

## 3. 新增模块

### 3.1 PTMDirectionMapper (`src/models/ptm_direction_mapper.py`)

**职责**: 将PTM修饰信息映射为DAVF所需的扰动输入格式。

**映射规则**:
| PTM类型 | 方向 | DAVF Direction | 生物学逻辑 |
|---------|------|----------------|-----------|
| Phosphorylation | activate | 2 (OE) | 磷酸化通常激活蛋白活性 |
| Acetylation | activate | 2 (OE) | 乙酰化通常激活转录 |
| Methylation | context-dependent | 1/2 | 取决于位点(K4me3=activate, K9me3=inhibit) |
| Ubiquitination | inhibit | 0 (KO) | 泛素化标记降解 |
| Sumoylation | inhibit | 0 (KO) | SUMO化通常抑制转录活性 |
| Succinylation | inhibit | 0 (KO) | 琥珀酰化抑制蛋白功能 |

**输入**: PTM类型列表 + 修饰位点蛋白名
**输出**: `(gene_ids: [B, K], directions: [B, K], attention_mask: [B, K])`

通过pathway_knowledge_base中的key_sites信息进一步细化方向判定。

### 3.2 DAVFInferenceModule (`src/models/davf_inference.py`)

**职责**: 封装DAVF/LatentDAVF的推理逻辑，冻结参数。

**核心功能**:
- 加载预训练checkpoint(从model_checkpoints.zip解压)
- 支持两种state_space: `gene`(DAVF) 和 `scvi_latent`(LatentDAVF)
- 提供`infer(gene_ids, directions, control_expression)` → `delta_expression`
- 冻结所有参数(推理模式)
- 可选解冻用于微调

### 3.3 DeltaProjection (`src/models/delta_projection.py`)

**职责**: 将DAVF输出的高维delta_expression压缩为固定维度特征。

**架构**:
```python
# delta_expression: [B, num_genes] (5000维)
# → projection: [B, 128]
DeltaProjection = nn.Sequential(
    nn.Linear(num_genes, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 128),
    nn.LayerNorm(128),
)
```

### 3.4 修改 PTM2CellNet (`src/models/architectures.py`)

在现有架构中增加DAVF分支:
- 新增`use_davf: bool`参数控制是否启用
- forward中并行执行序列编码和DAVF推理
- predictor输入维度从`embed_dim`变为`embed_dim + davf_feature_dim`

## 4. 文件搬入计划

从source_code.zip提取以下核心文件:
| 源文件 | 目标位置 | 说明 |
|--------|----------|------|
| `src/models/davf.py` | `src/models/davf.py` | DAVF核心模型 |
| `src/models/latent_davf.py` | `src/models/latent_davf.py` | LatentDAVF模型 |
| `src/models/biperturb.py` | `src/models/biperturb.py` | 方向编码器 |
| `src/models/attention.py` | `src/models/davf_attention.py` | DAVF注意力(避免与现有冲突) |
| `src/models/geneformer_embedding.py` | `src/models/geneformer_embedding.py` | 基因嵌入加载器 |
| `src/models/delta_predictor.py` | `src/models/delta_predictor.py` | Delta预测器 |
| `src/models/scvi_adapter.py` | `src/models/scvi_adapter.py` | scVI适配器 |
| `src/models/checkpoint_utils.py` | `src/models/davf_checkpoint_utils.py` | 检查点工具 |

## 5. 预训练模型使用

从model_checkpoints.zip解压到`checkpoints/`目录:
- `checkpoints/davf/model_a_finetuned/best_model.pt` → Gene空间DAVF(OE方向)
- `checkpoints/latent_davf_ibd_norman/best_model.pt` → Latent空间(Norman数据集)
- `checkpoints/latent_davf_ibd_replogle/best_model.pt` → Latent空间(Replogle数据集)
- `checkpoints/geneformer_embeddings.pt` → Geneformer预训练嵌入
- `checkpoints/scvi/` → scVI编解码器模型

**默认使用**: `latent_davf_ibd_norman` (10维latent, 与scVI配套)

## 6. 微调策略

**阶段1**: 冻结DAVF + 训练DeltaProjection + Predictor
- DAVF参数全部冻结
- 只训练DeltaProjection和更新后的Predictor
- 目标: 验证DAVF特征对细胞状态预测的贡献

**阶段2**: 端到端微调(可选)
- 解冻DAVF部分层(如velocity_field最后2层)
- 使用更小学习率(1e-5)
- 目标: 让DAVF适配PTM-specific分布

## 7. 配置接口

在`configs/`中新增配置:
```yaml
model:
  use_davf: true
  davf:
    state_space: "scvi_latent"  # "gene" | "scvi_latent"
    checkpoint_path: "checkpoints/latent_davf_ibd_norman/best_model.pt"
    scvi_model_path: "checkpoints/scvi/ibd_norman_model/model.pt"
    geneformer_path: "checkpoints/geneformer_embeddings.pt"
    feature_dim: 128
    freeze: true
    num_genes: 5000
```

## 8. 依赖新增

```
scvi-tools>=1.0  (用于scVI编解码)
anndata>=0.10    (scVI数据格式)
```

## 9. 验证标准

1. DAVF模型能正确加载checkpoint并推理
2. PTMDirectionMapper能将常见PTM修饰正确映射为direction
3. 级联融合后的模型在已有测试集上性能不降（基线对比）
4. DAVF特征对通路相关任务有正向贡献(通过ablation验证)
