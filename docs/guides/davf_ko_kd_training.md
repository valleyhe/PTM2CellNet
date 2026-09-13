# KO / KD 双 DAVF 模型训练与验证记录

**历史执行日期**：2026-09-02；**当前候选复训**：2026-09-04

本页记录 route-specific DAVF 的训练、坐标契约和工程验证。这里的真实 scPerturb
扰动数据、cell-baseline pair、checkpoint 与 bridge smoke 不能替代
`normal/disease` donor cohort，也不能单独构成正式生物学方向或效用验收。

## 结论

KO 和 KD 必须是两个独立的 DAVF 模型。它们不能共用一个方向标签，也不能把
Norman 数据集的历史 artifact 改名后充当 KO/KD checkpoint。当前已用本机真实
scPerturb 数据分别完成工程训练与桥接：

| 路径 | 数据 | 目标细胞 | 目标基因 | scVI | DAVF |
|---|---|---:|---:|---|---|
| KO | DixitRegev2016 K562，7d + 13d | 43,883 | 10 | `checkpoints/scvi/davf_ko_dixit` | `checkpoints/davf/davf_ko_dixit/best_model.pt` |
| KD | NadigOConner2024 Jurkat | 232,108 | 2,161 | `checkpoints/scvi/davf_kd_nadig` | `checkpoints/davf/davf_kd_nadig/best_model.pt` |

两条路径都固定为 `latent_dim=64`、`num_genes=4018`，使用当前验证过的
PerturbGen embedding asset，并且 embedding table 冻结。旧的
`configs/davf_integration.yaml` 保持历史默认值，不代表这两个新 checkpoint。

2026-09-04 重新构造了与 E2E 推理一致的 cell-baseline pair，并将工程候选模型写入
`*_cell_baseline_formal_20260904`；目录名中的 `formal` 是历史命名，不是生物学
验收状态。`configs/davf_ko.yaml` 和
`configs/davf_kd.yaml` 仍保留历史路径，直到候选模型完成独立验收。

本地 H5AD 的 `perturbation_type` 只记录为 `CRISPR`，没有细分机制；KO/KD 的
命名依据原始研究记录：Dixit K562 TF 数据的 GEO 设计明确为 CRISPR-KO，
[GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063)；
Jurkat 数据的 NCBI BioProject 明确为 genome-scale CRISPRi，
[PRJNA1049794](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1049794)。

## 数据契约

每条路径都执行以下不可省略的步骤：

1. 从真实 H5AD 读取 counts，保留明确的 control 和目标干预细胞。
2. 目标基因先用输入数据中的目标字段解析，再要求同时存在于当前 scVI 轴和
   PerturbGen vocabulary；不满足就排除，不能用随机 token 或零向量补齐。
3. 将目标基因轴固定为当前 scVI 的 4018 个有序 gene names。
4. 历史 pair 使用同 batch control 均值；当前正式候选 pair 使用同 batch、
   split-disjoint 的真实 control cell 作为 `z_0`，`z_1` 是干预细胞 latent，
   与 E2E 的具体 context cell 对齐。
5. 当前候选按 cell 划分 train/val/test，使已知 target 的细胞互斥；control
   cell 也在三个 split 间互斥，避免同一个 control 泄漏到验证集或测试集。
6. `gene_ids` 只表示 PerturbGen token row；解码表达列号只从
   `adapter.gene_names` 解析。

这些 pair 的 cell-level split 不等于 donor-held-out split；当前 scVI 在各自
prepared 数据全集上训练，`train_latent_davf.py` 也不会从 pair 自动推断 donor
身份。正式训练必须显式记录 train/held-out donor 列表，并由 manifest 证明来源
隔离；GSE 观测 cohort 与 DAVF 训练 donor 是否独立也不能凭同号方向推定。

方向编码固定为：`KO=0`、`KD=1`、`OE=2`。训练器和 checkpoint contract 都会
   检查方向类型，KO checkpoint 不能用 KD 输入推理。

## 可复现命令

以下命令在项目根目录执行。它们使用当前环境中的 GPU/CUDA；不覆盖已有目录时
会直接失败，避免无意替换模型。

### 1. 准备 KO H5AD

```bash
python scripts/prepare_davf_scperturb.py \
  --modality KO \
  --input data/raw/scperturb/DixitRegev2016_K562_TFs_7_days.h5ad \
  --input data/raw/scperturb/DixitRegev2016_K562_TFs_13_days.h5ad \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output data/processed/davf_scperturb/ko/prepared.h5ad
```

实际结果：`43,883 × 4,018`，10 个目标基因。

### 2. 准备 KD H5AD

```bash
python scripts/prepare_davf_scperturb.py \
  --modality KD \
  --input data/raw/scperturb/NadigOConner2024_jurkat.h5ad \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output data/processed/davf_scperturb/kd/prepared.h5ad
```

实际结果：`232,108 × 4,018`，2,161 个目标基因。

### 3. 分别训练 scVI

```bash
python scripts/train_davf_scvi.py \
  --input data/processed/davf_scperturb/ko/prepared.h5ad \
  --modality KO \
  --output checkpoints/scvi/davf_ko_dixit \
  --max-epochs 40 --batch-size 1024 --n-layers 2 \
  --seed 42 --accelerator gpu

python scripts/train_davf_scvi.py \
  --input data/processed/davf_scperturb/kd/prepared.h5ad \
  --modality KD \
  --output checkpoints/scvi/davf_kd_nadig \
  --max-epochs 40 --batch-size 1024 --n-layers 2 \
  --seed 42 --accelerator gpu
```

两个模型实际都保存为 `4018 × 64` 的 scVI 坐标，并通过
`ScVIAdapter` 重新加载检查。

### 4. 分别构造 latent pair

```bash
python scripts/build_davf_scperturb_pairs.py \
  --input data/processed/davf_scperturb/ko/prepared.h5ad \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --modality KO \
  --output-dir data/processed/davf_scperturb/ko/pairs \
  --split-strategy target --control-baseline mean \
  --seed 42 --max-cells-per-target 256 \
  --encoder-batch-size 1024 --device cuda

python scripts/build_davf_scperturb_pairs.py \
  --input data/processed/davf_scperturb/kd/prepared.h5ad \
  --scvi-model checkpoints/scvi/davf_kd_nadig \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --modality KD \
  --output-dir data/processed/davf_scperturb/kd/pairs \
  --split-strategy target --control-baseline mean \
  --seed 42 --max-cells-per-target 256 \
  --encoder-batch-size 1024 --device cuda
```

实际 pair 数量：

| 路径 | train | val | test | split 方向 |
|---|---:|---:|---:|---|
| KO | 2,048 | 256 | 256 | 三者均为 `0` |
| KD | 160,895 | 18,744 | 21,103 | 三者均为 `1` |

KO 的目标标签总数只有 10，按标签划分后验证和测试各只有一个目标标签，结果
只能作为工程基线；KD 的目标标签覆盖更充分。

上述 `ko/pairs`、`kd/pairs` 是历史 target-split/mean-baseline 记录。当前正式候选
使用以下 pair 目录：

| 路径 | train | val | test | baseline |
|---|---:|---:|---:|---|
| `ko_cell_baseline/pairs` | 2,560 | 2,524 | 2,522 | 同 batch 真实 control cell |
| `kd_cell_baseline/pairs` | 163,844 | 22,320 | 21,725 | 同 batch 真实 control cell |

### 5. 分别训练当前 LatentDAVF

```bash
python scripts/train_latent_davf.py \
  --train-data data/processed/davf_scperturb/ko_cell_baseline/pairs/train.npz \
  --val-data data/processed/davf_scperturb/ko_cell_baseline/pairs/val.npz \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --intervention-type KO \
  --output checkpoints/davf/davf_ko_cell_baseline_formal_20260904 \
  --epochs 40 --batch-size 2048 --learning-rate 1e-4 \
  --weight-decay 1e-4 --endpoint-loss-weight 1.0 \
  --direction-loss-weight 1.0 --patience 10 --num-workers 0 \
  --seed 42 --device cuda

python scripts/train_latent_davf.py \
  --train-data data/processed/davf_scperturb/kd_cell_baseline/pairs/train.npz \
  --val-data data/processed/davf_scperturb/kd_cell_baseline/pairs/val.npz \
  --scvi-model checkpoints/scvi/davf_kd_nadig \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --intervention-type KD \
  --output checkpoints/davf/davf_kd_cell_baseline_formal_20260904 \
  --epochs 40 --batch-size 2048 --learning-rate 1e-4 \
  --weight-decay 1e-4 --endpoint-loss-weight 1.0 \
  --direction-loss-weight 1.0 --patience 10 --num-workers 0 \
  --seed 42 --device cuda
```

实际训练结果：

| 路径 | 最优 epoch | 最优验证 flow loss | checkpoint |
|---|---:|---:|---|
| KO | 40 | endpoint `0.649685` | `checkpoints/davf/davf_ko_cell_baseline_formal_20260904/best_model.pt` |
| KD | 39 | endpoint `0.416126` | `checkpoints/davf/davf_kd_cell_baseline_formal_20260904/best_model.pt` |

flow loss 只能说明训练目标有数值收敛，不能替代 held-out 基因方向准确率。

当前入口已接受 `--train-donors` / `--held-out-donors`（及可选
`--frozen-cohort-manifest`）。上表命令是 2026-09-04 的历史复训记录，当时未绑定
donor split；不能把这些 checkpoint 的验证 loss 解释为 held-out donor 未泄漏，也
不能把它们称为 formal biological PASS。
新的正式训练应同时写出 `training_metrics.json` 中的 `donor_split`。

## 验证

先验证 checkpoint、scVI 和 embedding 三者的精确契约：

```bash
python scripts/validate_latent_davf_checkpoint.py \
  --checkpoint checkpoints/davf/davf_ko_cell_baseline_formal_20260904/best_model.pt \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --intervention-type KO \
  --device cpu

python scripts/validate_latent_davf_checkpoint.py \
  --checkpoint checkpoints/davf/davf_kd_cell_baseline_formal_20260904/best_model.pt \
  --scvi-model checkpoints/scvi/davf_kd_nadig \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --intervention-type KD \
  --device cpu
```

两条命令实际均返回 `ok=true`、`schema_version=2`、`model_type=LatentDAVF`、
`latent_dim=64`、`num_genes=4018`，并报告对应的 `intervention_type` 与
`direction_code`。

### Held-out endpoint 评估

训练集和验证集 loss 不能证明 DAVF 在未见样本上的终点预测有效。因此使用独立
test NPZ 做 ODE endpoint 评估。评估器会再次严格检查 checkpoint、scVI、embedding
和干预类型，并同时报告 identity baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_latent_davf.py \
  --test-data data/processed/davf_scperturb/ko_cell_baseline/pairs/test.npz \
  --checkpoint checkpoints/davf/davf_ko_cell_baseline_formal_20260904/best_model.pt \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --intervention-type KO \
  --output outputs/davf_evaluation/ko_test.json \
  --batch-size 2048 --num-steps 50 --device cuda

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_latent_davf.py \
  --test-data data/processed/davf_scperturb/kd_cell_baseline/pairs/test.npz \
  --checkpoint checkpoints/davf/davf_kd_cell_baseline_formal_20260904/best_model.pt \
  --scvi-model checkpoints/scvi/davf_kd_nadig \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --intervention-type KD \
  --output outputs/davf_evaluation/kd_test.json \
  --batch-size 2048 --num-steps 50 --device cuda
```

`relative_to_identity_mse < 1` 只表示 DAVF latent endpoint 比不扰动基线更接近观测
`z_1`。当前评估器还会读取 pair provenance 中的真实 prepared H5AD，用
`adapter.gene_names` 将 PerturbGen token 解析为 scVI decoder 列，再报告
`gene_direction_sign_accuracy` 和 `gene_direction_inconclusive_fraction`；只有这一步
才是在验证目标基因的表达变化方向。它仍不能单独构成生物学结论。

当前正式候选实测 gene-level 结果：

| 路径 | 1-step sign accuracy | 50-step sign accuracy | inconclusive |
|---|---:|---:|---:|
| KO | 0.6749 | 0.6657 | 0 |
| KD | 0.7088 | 0.5785 | 0 |

KD 的多步结果明显下降，因此下游 direction gate 当前优先使用 `num_steps=1`；50-step
只作为积分稳定性诊断，除非完成覆盖积分轨迹的重新训练。

### 当前候选配置与 E2E 入口

两条正式候选现在有独立的 dated 配置：

| 路径 | 配置 | checkpoint | 推理步数 |
|---|---|---|---:|
| KO | `configs/davf_ko_cell_baseline_formal_20260904.yaml` | `davf_ko_cell_baseline_formal_20260904/best_model.pt` | 1 |
| KD | `configs/davf_kd_cell_baseline_formal_20260904.yaml` | `davf_kd_cell_baseline_formal_20260904/best_model.pt` | 1 |

配置中的 `provenance` 记录 pair、prepared H5AD、cell split 和 cell baseline；运行时
真正消费的 `davf` 段仍严格固定 `latent_dim=64`、`num_genes=4018`、对应 scVI
模型和同一个 PerturbGen embedding asset。历史 `configs/davf_ko.yaml`、
`configs/davf_kd.yaml` 没有切换。

真实资产 CLI 回归测试命令：

```bash
CUDA_VISIBLE_DEVICES=0 PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 \
pytest -q tests/real_assets/test_real_davf_perturbgen_bridge.py
```

该测试使用两份真实 prepared H5AD，验证候选 checkpoint 能完成 scVI 编码、DAVF
预测、scVI 解码和方向 gate，并检查结果中的 PerturbGen token route。它不自动运行
六阶段 PerturbGen，避免把验证入口和长时间训练混在一起。

本机已有一条真实六阶段 PerturbGen 工程链：
`outputs/perturbgen/local_adaptation_datlinger2021_v10/` 的 tokenise、train-mask、
train-decoder、perturb、export_gene_embeddings 和 report manifest 都是
`success`。其中 mask/decoder 只在 750-cell Datlinger-Bock 子集上训练 2 个 epoch，
因此这些权重目前只能证明 stage wiring、checkpoint 传递和输出 schema 可用，不能
作为正式全规模效用模型。正式 E2E 仍需要在满足 donor/state 契约的数据上重新运行
六阶段，并在通过 DAVF gate 后按候选的独立输出目录执行。六阶段中的
`report` 只汇总 manifest，不是生物学 PASS。

Workflow A 的研究目标是固定一次 cohort、token vocabulary、训练配置和资产版本，
候选阶段只运行 `source_intervention=[src]` 与 `within_state=[tgt]+pert_tps`
两条扰动和后续统计；当前 orchestrator 仍会对每个通过 gate 的候选重复
`tokenise/train_mask/train_decoder/export_gene_embeddings/report`，没有跨候选 reuse
CLI。Workflow B 是独立的“基础 encoder 静态冻结 embedding 资产 → LatentDAVF 重训
→ Gate-E”生命周期；export 使用基础 encoder checkpoint，不读取候选阶段 checkpoint，
也不回灌当前 DAVF。

还必须验证新进程直接 decode：scVI checkpoint 没有嵌入 AnnData 时，调用方仍需
传入含完整 4018 gene axis 和训练协变量的真实 AnnData；adapter 会重新绑定它，
不会猜测 batch 或 gene order。KO、KD 两个真实模型的 `[1,64] → [1,4018]`
decode 已实际通过，输出全为 finite。

只有通过三方方向 gate 的 candidate 才能形成 PerturbGen invocation；默认 E2E 只
生成 gate/invocation，必须显式加 `--run-perturbgen` 才调用外部六阶段。DAVF 只
提供方向和置信证据，PerturbGen 负责全基因表达效用；当前的同号 gate 尚未定义
context、干预、对比基准和“病程关联/复现/逆转”目标，不能把它写成科学因果 gate。
当前默认配置仍指向历史 checkpoint，因此候选验收阶段应显式传入新 checkpoint；
推理时不能把两条路径的方向标签混用。

## 当前边界

- 新 checkpoint 已是真实数据上的可运行训练产物，并通过 endpoint/方向/bridge
  工程验收；仍不等于已通过生物学发布门。还需要真实 normal/disease raw counts、
  显式 donor、至少 3 个共享 donor、canonical Ensembl、冻结 embedding/manifest、
  可追溯 donor split、独立 held-out target、至少三个 seed，以及 scVI decode 后的
  目标基因 sign accuracy、相关性和接近零变化的 inconclusive 比例。
- KO 训练数据的目标标签数量较少，不能据此声称对全部 KO 泛化。
- KO 与 KD 使用不同细胞系和不同 scVI 坐标系，因此两个 DAVF 的 latent 数值
  不能直接比较；合并点应是方向证据进入 PerturbGen，而不是拼接 latent。
- 当前 scVI 是在各自 prepared 全体细胞上训练的，cell split 属于 transductive
  latent evaluation；donor-held-out 研究需要按 donor 重训 scVI 并重新生成 pair。
  GSE donor 与 DAVF 训练 donor 的独立性需由来源 manifest 证明，当前训练入口不会
  自动验证该关系。
