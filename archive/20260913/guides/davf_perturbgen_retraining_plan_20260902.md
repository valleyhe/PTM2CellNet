# DAVF 与 PerturbGen 重训练/微调执行方案

**日期**：2026-09-02  
**结论**：DAVF 重训练，PerturbGen 基于公开 encoder 权重微调；两者保持独立训练、独立验证，最后通过严格的方向门连接。

> **状态说明**：本文记录的是早期 Norman/通用方案。当前 KO 与 KD 已改为使用
> 独立 scPerturb 数据、独立 scVI 坐标和独立 checkpoint；可直接复现的实际命令
> 与结果见 [`davf_ko_kd_training.md`](davf_ko_kd_training.md)。本文中的旧
> Norman artifact 不得当作当前 KO/KD 正式模型。

## 1. 从第一性原理确定分工

目标不是让两个模型都预测同一个结果，而是把一条可审计的因果链拆成两个问题：

```text
PTM site + gene
      │
      ├─ DAVF：z0 → z1，验证目标基因表达变化方向
      │          └─ scVI decode → target Δexpression → up/down
      │
      └─ PerturbGen：给定干预动作，预测全基因表达效用
                   └─ count/embedding 结果 → rescue、signature、候选排序
```

这要求三个坐标系不能混淆：

| 对象 | 真实含义 | 允许的索引来源 |
|---|---|---|
| DAVF `gene_ids` | PerturbGen gene-token row，供条件编码器读取 embedding | verified PerturbGen `vocabulary.json` |
| DAVF `z_0/z_1` | 当前 scVI 模型的 latent 坐标 | 当前 scVI encoder，固定 64 维 |
| scVI target gene index | 解码表达矩阵的列号 | `adapter.gene_names` 的有序列表 |

因此，PerturbGen token `328` 不能被当作 scVI 的第 328 个 gene；当前真实 LCK 例子中，PerturbGen token 是 `328`，scVI decoder row 是 `113`。代码已经把这两个索引放在不同入口并加入真实集成测试。

## 2. 当前本机基线与选择

| 资产 | 当前事实 | 决策 |
|---|---|---|
| scVI | `checkpoints/scvi/ibd_norman_model`，4018 genes × 64 latent；使用 `batch`、`dataset` 协变量 | 固定为 DAVF 的唯一 latent 坐标系 |
| PerturbGen embedding | `outputs/perturbgen/embedding_asset_20260822`，18967 × 768，Ensembl vocabulary，manifest 已校验 | 注入 DAVF，embedding table 冻结 |
| Norman GSE133344 | 111445 个有身份细胞，69,686 个 control，131 个扰动标签；按 `gemgroup` 匹配 control | 先用于 DAVF 可运行重训练；不能单独宣称跨数据集泛化 |
| 旧 DAVF checkpoint | 没有当前 `LatentDAVF.predict()`，不能作正式方向证据 | 不做旧权重微调，直接用当前架构重训练 |
| PerturbGen foundation | 本机已有公开 masking checkpoint；架构为 768 维 encoder | 不从零训练，做 masking 适配，再训练 count decoder |
| Datlinger 本地 smoke | 750 cells、25904 原始 genes，400 control/350 LCK；只适合工程验证 | 只验证六阶段 wiring，不作为生物学结论 |

核心判断是：DAVF 的旧权重连模型接口都不满足，微调没有意义；PerturbGen 的公开 encoder 已提供通用基座，数据量有限时从零训练反而丢掉预训练信息。PerturbGen 的 count decoder 不是当前本机已确认的公开成品，因此必须在目标数据上训练/微调。

## 3. DAVF：正式重训练方案

### 3.1 输入构造

`build_davf_latent_pairs.py` 做以下严格处理：

1. 读取 Norman 10x matrix、gene table、barcode 和 identity。
2. 只保留有明确扰动标签或 control 标签的细胞。
3. 每个扰动细胞只从同一 `gemgroup` 的 control 中抽取 `z_0`，扰动细胞自身的 scVI latent 是 `z_1`。
4. 用当前 `adapter.gene_names` 对原始矩阵重新排列；重复 symbol 列相加，不能随便选一列。
5. 扰动标签按 label 划分 train/val/test，不能把同一扰动的细胞拆到不同集合。
6. `gene_ids` 只通过 PerturbGen asset 的 Ensembl token 解析；没有 token 或目标基因不可观测时记录排除原因并停止伪造样本。
7. 当前 Norman 标签全部按 KO 编码（`direction=0`）；没有真实 OE/KD 监督时不宣称已训练出 OE/KD 能力。

执行命令：

```bash
python scripts/build_davf_latent_pairs.py \
  --geo-root data/raw/norman_adamson \
  --scvi-model checkpoints/scvi/ibd_norman_model \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output-dir data/processed/davf_latent/norman_gse133344 \
  --device cuda --seed 42
```

当前实际产物：

| split | 样本数 | latent shape | active target 数 |
|---|---:|---|---:|
| train | 46128 | `[46128, 64]` | 1–2 |
| val | 5349 | `[5349, 64]` | 1–2 |
| test | 7006 | `[7006, 64]` | 1–2 |

有 3 个标签因目标基因不在可观测 raw gene table 中被明确排除：`C19orf26`、`C3orf72_FOXL2`、`TGFBR2_C19orf26`。

### 3.2 模型与训练

训练的是当前 ODE/flow-matching `LatentDAVF`，不是旧 `delta_mlp` 模型：

| 参数 | 当前执行值 | 原因 |
|---|---:|---|
| latent dim | 64 | 与 scVI checkpoint 完全一致 |
| decoder gene count | 4018 | 与 scVI `gene_names` 完全一致 |
| embedding | 18967 × 768 | 与 verified PerturbGen asset 完全一致 |
| embedding table | frozen | 不让 DAVF 改变 token row 语义 |
| batch size | 512 | 当前 P40 显存下的实际配置 |
| learning rate | `1e-4` | flow field 主参数训练 |
| weight decay | `1e-4` | 控制小样本/标签不平衡下的过拟合 |
| gradient clip | `1.0` | 防止 latent velocity 数值爆炸 |
| epoch/patience | 30 / 5 | 先得到可运行 current checkpoint |

```bash
python scripts/train_latent_davf.py \
  --train-data data/processed/davf_latent/norman_gse133344/train.npz \
  --val-data data/processed/davf_latent/norman_gse133344/val.npz \
  --scvi-model checkpoints/scvi/ibd_norman_model \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output checkpoints/davf/latent_davf_perturbgen_4018 \
  --epochs 30 --batch-size 512 --learning-rate 1e-4 \
  --weight-decay 1e-4 --patience 5 --device cuda --seed 42
```

实际输出：best epoch `29`，validation flow loss `0.49508876927791745`，正式 checkpoint 为：

```text
checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt
```

这个 flow loss 只证明训练目标收敛到一个有限值，不等于生物学方向准确率。正式方向门还必须看 held-out expression direction。

### 3.3 当前验证顺序

```bash
python scripts/validate_latent_davf_checkpoint.py \
  --checkpoint checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt \
  --scvi-model checkpoints/scvi/ibd_norman_model \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822
```

当前实际结果：`ok=true`、`schema_version=2`、`model_type=LatentDAVF`、`latent_dim=64`、`num_genes=4018`、asset shape `[18967,768]`、scVI shape `[4018,64]`。

正式推理必须执行：

```python
adapter = ScVIAdapter.from_trained_model(
    "checkpoints/scvi/ibd_norman_model",
    config=ScVIAdapterConfig(
        model_path="checkpoints/scvi/ibd_norman_model",
        n_latent=64,
        device="cuda",
    ),
)
davf = DAVFInferenceModule(
    DAVFInferenceConfig(
        state_space="scvi_latent",
        checkpoint_path="checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt",
        scvi_model_path="checkpoints/scvi/ibd_norman_model",
        embedding_asset_path="outputs/perturbgen/embedding_asset_20260822",
        gene_names_path="data/raw/norman_adamson/GSE133344/GSE133344_filtered_genes.tsv.gz",
        latent_dim=64,
        num_genes=4018,
        device="cuda",
    )
)
davf.bind_scvi_adapter(adapter)
mapper = davf.build_perturbgen_direction_mapper()
mapper_output = mapper.map_batch(
    [[{"type": "phosphorylation"}]],
    [["LCK"]],                       # API symbol → verified asset token
)
scvi_index = adapter.resolve_target_gene_indices(["LCK"])[0]  # decoder row
evidence = davf.predict_expression_direction(
    mapper_output,
    z_0,
    target_gene_symbols=["LCK"],
    target_ensembl_ids=["ENSG00000182866"],
    scvi_context=adata_with_correct_covariates,
)
```

上面 `adata_with_correct_covariates` 代表真实输入细胞的 AnnData（必须含 scVI 训练所需协变量）；完整可运行验证已经固化在 `tests/integration/test_scvi_davf_connection.py`。实际测试确认：LCK token 与 scVI row 分离，输出 delta finite，`model_source="davf"`。

### 3.4 DAVF 正式发布门

当前只通过了“资产/接口/数值可运行”门，尚未通过生物学门。正式重训练需增加：

| 检查 | 具体要求 |
|---|---|
| latent transition | test split 的 latent MSE、cosine、每个扰动 label 的误差分布 |
| gene direction | scVI decode 后目标基因 delta 的 sign accuracy、Spearman/线性相关 |
| 置信度 | 多次采样或 bootstrap 的方向稳定性；接近 0 的 delta 记为 inconclusive |
| 泛化 | 未见扰动 label、未见 gemgroup/donor 的独立测试；不能只测训练标签 |
| 重复性 | 至少 3 个 seed；所有 seed 都要保存 checkpoint 与 manifest |
| 模式边界 | 当前 KO 数据只验 KO；OE/KD 需要显式监督数据或单独标记为未验证 |

任一目标 token 无法解析、`z_1` 非有限、decoder gene order 不一致，直接失败；不能把它变成零向量或随机 embedding。

## 4. PerturbGen：公开权重微调方案

### 4.1 为什么选择微调

PerturbGen 的任务是从干预后的 token 序列预测完整表达状态。其预训练 masking encoder 已学习 gene-token 表示，但目标数据的细胞类型、测序深度和扰动分布仍不同，所以最稳的路径是：

```text
公开 masking encoder
        ↓
目标数据 tokenise
        ↓
train_mask：小学习率适配遮挡恢复
        ↓
train_decoder：接入 count decoder，ZINB 训练
        ↓
src/tgt 扰动推理
```

当前不直接声称“公开 checkpoint 可零样本完成六阶段推理”：本机实测表明 count decoder checkpoint 需要与目标 tokenized axis、mapping、模型维度一致，不能只把一个 encoder checkpoint 填到 `val.py`。

### 4.2 六阶段执行契约

入口配置：[`configs/integration/perturbgen_local_adaptation.yaml`](../../configs/integration/perturbgen_local_adaptation.yaml)。每一阶段都写 `stage_manifest.json`，下游只接受上游 manifest 指定的单一产物。

| 阶段 | 当前 smoke 设置 | 正式设置原则 | 必须检查 |
|---|---|---|---|
| `tokenise` | raw h5ad；HVG after tokenisation；2000 HVG；强制保留 LCK；`exclude_non_GF_genes=true` | 用正式 raw counts 和固定目标 gene 列表；目标必须在 token axis | raw counts、ENSG、target coverage、token axis |
| `train_mask` | foundation encoder；batch 16；2 epochs；`5e-5`；wd `1e-6` | 从 10–30 epochs 起，按 held-out loss 早停；不改 token vocabulary | 单一 masking checkpoint、无多 checkpoint 歧义 |
| `train_decoder` | batch 8；2 epochs；`5e-5`；wd `0.01`；`loss_mode=zinb` | 保持 `d_model=768`、8 heads、6 layers；按 held-out count loss 早停 | decoder 与 masking checkpoint 的维度/轴一致 |
| `perturb` | `mask`、`src`、LCK、`pred_tps=1`、`n_samples=1` | `src` 与 `tgt` 分开跑；mask 为主，pad/delete/overexpress 做敏感性 | h5ad shape、finite、true/pred layers、embedding fields |
| `export_gene_embeddings` | 从同一 foundation checkpoint 导出 18967 × 768 asset | 若导出微调后的 embedding，必须重新做 manifest 和 DAVF 重训 | tensor key、词表行号、shape、hash |
| `report` | 汇总所有 stage manifest | 正式报告加入 donor/seed/null/FDR 与代码/权重 provenance | 不能用 smoke 报告冒充正式效用 |

运行命令：

```bash
TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 WANDB_MODE=offline \
PTM2CELLNET_PERTURBGEN_PYTHON=/home/scu/anaconda3/envs/perturbgen/bin/python \
PTM2CELLNET_PERTURBGEN_LOCAL_OUTPUT_ROOT=/home/scu/PTM2CellNet/outputs/perturbgen/local_adaptation_datlinger2021_v10 \
PTM2CELLNET_PERTURBGEN_LOCAL_DATASET=datlinger2021_local_adaptation_v10 \
PTM2CELLNET_PERTURBGEN_LOCAL_INPUT_H5AD=/home/scu/PTM2CellNet/ref/Perturbgen-src/data/perturbgen_m0_smoke/datlinger2021_m0smoke.h5ad \
PTM2CELLNET_PERTURBGEN_LOCAL_ENCODER_CKPT=/home/scu/PTM2CellNet/perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt \
PTM2CELLNET_PERTURBGEN_LOCAL_GENE_MEDIAN=/home/scu/PTM2CellNet/ref/Perturbgen-src/perturbgen/pp/gene_median_dict_gftokens_gc95M.pkl \
PTM2CELLNET_PERTURBGEN_LOCAL_TOKEN_DICT=/home/scu/PTM2CellNet/ref/Perturbgen-src/perturbgen/pp/token_dict_gftokens_gc95M.pkl \
PTM2CELLNET_PERTURBGEN_LOCAL_GENE_MAPPING=/home/scu/PTM2CellNet/ref/Perturbgen-src/perturbgen/pp/ensembl_mapping_dict_gc95M.pkl \
PTM2CELLNET_PERTURBGEN_LOCAL_TARGETS=/home/scu/PTM2CellNet/configs/integration/perturbgen_targets/lck.csv \
python scripts/run_perturbgen_pipeline.py \
  --config configs/integration/perturbgen_local_adaptation.yaml \
  --resume --gpu-lock-file outputs/perturbgen/.gpu.lock
```

当前六阶段真实运行成功，输出根目录：

```text
outputs/perturbgen/local_adaptation_datlinger2021_v10/
```

实际关键结果：

| 产物 | 结果 |
|---|---|
| masking checkpoint | `train_mask/model/checkpoints/*epoch=01.ckpt` |
| count decoder checkpoint | `train_decoder/model/checkpoints/*epoch=01.ckpt` |
| perturb h5ad | `perturb/results/20260902-15:55_minference_adata_gLCK_ssrc_tmask.h5ad` |
| h5ad shape | `227 × 2001` |
| h5ad layers | `true_counts`、`pred_counts` |
| h5ad embeddings | `true_cls`、`perturbed_cls`、`mean_cos_similarity`、`gene_cos_similarity` |
| exported asset | `export_gene_embeddings/asset`，`18967 × 768` |
| report | `report/pipeline_report.json` |

此前真实执行暴露并已修复的连接错误包括：默认不存在的 `cell_type_cellgen_harm`、numpy index truth 判断、LCK 被 HVG 丢弃、配置缺少 `d_model`、以及 25904 原始 gene 与 2001 tokenized gene 轴不一致。配置现在明确 `exclude_non_GF_genes=true`，不再用错误 shape 的 h5ad 继续推理。

### 4.3 PerturbGen 正式效用门

Datlinger 750-cell smoke 只能证明代码能运行。正式效用训练必须换成通过数据审计的队列，并且：

1. 训练/验证按 donor 或实验批次隔离；同一 donor 的细胞不能同时泄漏到 train 和 test。
2. `src`：从 reference/control 状态施加候选干预，检查是否接近 disease signature。
3. `tgt`：在 disease 状态内部评估干预的稳定性，不能用 `src` 的结果代替。
4. 每条路径至少 3 个可评估 donor、3 个 seed；正式 null 至少 99 个匹配扰动，并做经验 p 值和 BH-FDR。
5. 目标 gene 本身从 rescue signature 中排除，避免模型仅靠“把目标改掉”获得假阳性。
6. `up/down` 方向不由 `KO/KD/OE` 数字码直接推导；动作码和表达方向是两个字段。

## 5. 两条路径如何连接

正式候选顺序固定为：

```text
PTM proposal
   ↓
DAVF + scVI：输出 target Δexpression 的 up/down evidence
   ↓ 与独立 observed direction 比较
通过三方方向 gate
   ↓
PerturbGen：按 observed direction 选择 KO/OE 等 corrective action
   ↓
source_intervention AND within_state
   ↓
PASS / FAIL / INCONCLUSIVE
```

这里 `PTMDirectionMapper` 的 phosphorylation→OE 只是 PTM 的动作先验，不是实测表达方向。当前 `src/integration/perturbgen/mainline.py` 已强制：缺 DAVF 正式 evidence 或方向不一致时，不进入 PerturbGen 效用评估；`scripts/run_davf_perturbgen_e2e.py` 已把这条 gate 接到 PerturbGen stage plan。底层 `run_perturbgen_pipeline.py` 仍是有意保留的纯 runner 入口，直接调用它不会获得 DAVF gate 证据。

## 6. 当前完成度与下一步

| 内容 | 状态 |
|---|---|
| 当前 scVI 连接、gene order、covariate decode | 已完成并通过真实本地测试 |
| current DAVF schema-v2 checkpoint | 已完成；真实验证通过 |
| Norman latent pair | 已完成；按 label split，KO-only |
| PerturbGen 六阶段本地微调/推理/export/report | 已完成 smoke；真实 h5ad schema 通过 |
| symbol→ENSG→PerturbGen token | 已完成；歧义 symbol 只屏蔽，不猜测 |
| DAVF held-out 生物学方向准确率 | 未完成，当前 flow loss 不能替代它 |
| PerturbGen 正式 donor 队列与双路径科学门 | 未完成，smoke 数据不具备 donor 证据 |
| DAVF→方向 gate→PerturbGen 自动接线 | 已完成；真实资产桥接通过，正式 donor/效用门仍未完成 |

下一次正式运行的最优顺序：

1. 先为目标 PerturbGen 队列建立 raw counts/ENSG/donor/state/cell type 审计；不合格数据不进入训练。
2. 如果训练多个研究的联合 scVI，先重新训练一个统一 4018-gene、64-latent scVI，再重新生成 DAVF pairs；不能把不同 scVI latent 直接拼接。
3. 用正式队列对 PerturbGen 做 masking/count 微调，固定 vocabulary 和 gene axis，跑 `src`、`tgt` 两路。
4. DAVF 与 PerturbGen 都用 held-out、3 seeds 和独立 null 验证后，才把 E2E CLI 报告升级为正式 release evidence。

## 7. 可复核入口

- DAVF pair builder：[`scripts/build_davf_latent_pairs.py`](../../scripts/build_davf_latent_pairs.py)
- DAVF trainer：[`scripts/train_latent_davf.py`](../../scripts/train_latent_davf.py)
- DAVF contract validator：[`scripts/validate_latent_davf_checkpoint.py`](../../scripts/validate_latent_davf_checkpoint.py)
- PerturbGen local config：[`configs/integration/perturbgen_local_adaptation.yaml`](../../configs/integration/perturbgen_local_adaptation.yaml)
- 当前 DAVF/scVI/PerturbGen 真实集成测试：[`tests/integration/test_scvi_davf_connection.py`](../../tests/integration/test_scvi_davf_connection.py)
- DAVF→PerturbGen E2E 编排器：[`src/integration/perturbgen/orchestrator.py`](../../src/integration/perturbgen/orchestrator.py)
- DAVF→PerturbGen E2E CLI：[`scripts/run_davf_perturbgen_e2e.py`](../../scripts/run_davf_perturbgen_e2e.py)
- KO/KD 报告合并：[`scripts/merge_davf_perturbgen_reports.py`](../../scripts/merge_davf_perturbgen_reports.py)
- 当前运行报告：[`outputs/perturbgen/local_adaptation_datlinger2021_v10/report/pipeline_report.json`](../../outputs/perturbgen/local_adaptation_datlinger2021_v10/report/pipeline_report.json)
