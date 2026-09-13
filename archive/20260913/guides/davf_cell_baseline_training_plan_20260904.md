# DAVF × PerturbGen 双路径训练与验证计划（2026-09-04）

## 1. 目标

主线是：

```text
真实 scVI context cell z0
        + PerturbGen gene token
        + KO/KD direction
        ↓
route-specific LatentDAVF
        ↓
预测 latent endpoint z1
        ↓
scVI decoder（gene_names 来自 adapter.gene_names）
        ↓
DAVF 只验证/筛选表达变化方向
        ↓
通过 gate 的候选进入 PerturbGen 六阶段流程
```

DAVF 和 PerturbGen 不承担同一个任务：DAVF 负责方向证据，PerturbGen 负责后续表达效用预测。因此 KO、KD 必须分别训练和推理，不能用一个 checkpoint 混合两种方向。

## 2. 第一性原理决策

DAVF 的真实推理函数是 `z1 = f(z0, gene_token, direction)`，其中 `z0` 是用户 context cell 的 scVI latent。训练 pair 的 `z_0` 必须来自同一类真实 control cell，且和 target cell 保持相同 batch；否则训练看到的是“control 均值到 target”的任务，推理执行的是“具体 context cell 到 target”的任务，边界条件不一致。

因此正式 E2E 路径采用：

| 项目 | 正式候选设置 | 原因 |
|---|---|---|
| pair baseline | 同 batch、split-disjoint 的真实 control cell | 与 E2E 的具体 context cell 对齐 |
| split | `cell` | 同一 target 的不同细胞进入不同 split，验证已知 target 的细胞泛化 |
| target split | `target` 仅作为严格的未见 target 外推实验 | 当前 KO/KD target 数量和 scVI 训练口径不适合冒充生产准入 |
| DAVF token | `PerturbGen embedding asset` 的 gene-token index | 只供 DAVF 条件编码 |
| scVI gene index | 运行时从 `adapter.gene_names` 解析 | 禁止把 PerturbGen token 当 decoder column |
| checkpoint | KO/KD 各自 schema-v2 `LatentDAVF` | 旧 checkpoint 没有当前 `predict()`，不能产生正式方向证据 |

## 3. 代码修改

| 文件 | 修改内容 | 验证 |
|---|---|---|
| `src/data/davf_scperturb.py` | 增加 target-cell split；增加 `control_baseline=cell`；输出 `pair_control_cell_ids/pair_control_batches`；保留 `mean` 作为历史实验设置 | split 单元测试；真实 KO/KD pair 契约检查 |
| `scripts/build_davf_scperturb_pairs.py` | 暴露 `--split-strategy` 和 `--control-baseline` | CLI/真实数据重建 |
| `scripts/train_latent_davf.py` | 在 flow matching 之外增加精确 `t=0` endpoint loss 和 latent direction cosine loss；训练/导出时强制单一 KO/KD/OE 方向 | 训练器单元测试；GPU smoke；checkpoint 重载 |
| `src/models/davf_checkpoint_contract.py` | schema-v2 checkpoint 支持 expected intervention type 校验 | 方向不匹配回归测试 |
| `scripts/evaluate_latent_davf.py` | 增加 held-out endpoint、identity baseline、latent 指标，以及用真实 H5AD 解码目标 gene 的 endpoint/sign/inconclusive 指标 | CPU 单元测试；GPU test split 评估 |
| `configs/davf_ko_cell_baseline_formal_20260904.yaml` | 独立的 KO 候选推理配置，固定当前 checkpoint/scVI/embedding 和 `num_steps=1` | 真实 KO CLI 回归测试 |
| `configs/davf_kd_cell_baseline_formal_20260904.yaml` | 独立的 KD 候选推理配置，固定当前 checkpoint/scVI/embedding 和 `num_steps=1` | 真实 KD CLI 回归测试 |
| `tests/unit/...` | 覆盖 pair split、训练目标、checkpoint 合同、评估指标和 dated route 配置 | 50 个相关测试 |

训练目标为：

```text
flow_loss = MSE(v_t, z1 - z0), t ~ Uniform(0, 1)
endpoint_loss = MSE(v_0, z1 - z0)
direction_loss = 1 - cosine(v_0, z1 - z0)
total = flow_loss + endpoint_weight * endpoint_loss
                 + direction_weight * direction_loss
```

`endpoint_loss` 是必要的，因为生产预测从 `t=0` 开始；只优化随机内部时间不能保证初始速度可用。

## 4. 数据和训练命令

当前正式候选资产：

| 路由 | prepared AnnData | scVI | pair 输出 |
|---|---|---|---|
| KO | `data/processed/davf_scperturb/ko/prepared.h5ad` | `checkpoints/scvi/davf_ko_dixit` | `data/processed/davf_scperturb/ko_cell_baseline/pairs` |
| KD | `data/processed/davf_scperturb/kd/prepared.h5ad` | `checkpoints/scvi/davf_kd_nadig` | `data/processed/davf_scperturb/kd_cell_baseline/pairs` |

如需从原始 prepared 数据重新生成 pair：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_davf_scperturb_pairs.py \
  --input data/processed/davf_scperturb/ko/prepared.h5ad \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --modality KO \
  --output-dir data/processed/davf_scperturb/ko_cell_baseline/pairs \
  --split-strategy cell \
  --control-baseline cell \
  --seed 42 --max-cells-per-target 256 \
  --encoder-batch-size 1024 --device cuda

CUDA_VISIBLE_DEVICES=0 python scripts/build_davf_scperturb_pairs.py \
  --input data/processed/davf_scperturb/kd/prepared.h5ad \
  --scvi-model checkpoints/scvi/davf_kd_nadig \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --modality KD \
  --output-dir data/processed/davf_scperturb/kd_cell_baseline/pairs \
  --split-strategy cell \
  --control-baseline cell \
  --seed 42 --max-cells-per-target 256 \
  --encoder-batch-size 1024 --device cuda
```

正式候选 checkpoint 不覆盖历史默认配置，先写入带日期的独立目录：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_latent_davf.py \
  --train-data data/processed/davf_scperturb/ko_cell_baseline/pairs/train.npz \
  --val-data data/processed/davf_scperturb/ko_cell_baseline/pairs/val.npz \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --intervention-type KO \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output checkpoints/davf/davf_ko_cell_baseline_formal_20260904 \
  --epochs 40 --batch-size 2048 --learning-rate 1e-4 \
  --endpoint-loss-weight 1.0 --direction-loss-weight 1.0 \
  --patience 10 --num-workers 0 --seed 42 --device cuda

CUDA_VISIBLE_DEVICES=0 python scripts/train_latent_davf.py \
  --train-data data/processed/davf_scperturb/kd_cell_baseline/pairs/train.npz \
  --val-data data/processed/davf_scperturb/kd_cell_baseline/pairs/val.npz \
  --scvi-model checkpoints/scvi/davf_kd_nadig \
  --intervention-type KD \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output checkpoints/davf/davf_kd_cell_baseline_formal_20260904 \
  --epochs 40 --batch-size 2048 --learning-rate 1e-4 \
  --endpoint-loss-weight 1.0 --direction-loss-weight 1.0 \
  --patience 10 --num-workers 0 --seed 42 --device cuda
```

## 5. 验证顺序和准入条件

1. `load_latent_davf_pairs` 必须通过 scVI 路径、ordered gene names、PerturbGen manifest、token 范围和方向码校验。
2. 训练结束后必须能用 `load_current_davf_checkpoint` 严格重载，且 checkpoint 的 `intervention_type` 与路由一致。
3. held-out test 同时运行 `--num-steps 1` 和 `--num-steps 50`。最低工程准入是 endpoint MSE 优于 identity baseline（`relative_to_identity_mse < 1`），并且真实解码目标 gene 的 `gene_direction_sign_accuracy` 高于随机基线、报告 `gene_direction_inconclusive_fraction`；latent cosine/sign 和预测幅度作为辅助指标。这不是生物学有效性的充分证明。
4. 使用真实 context cell 运行 DAVF→PerturbGen bridge。必须得到 `pass` 或明确的方向 gate 拒绝，不能产生 zero fallback；PerturbGen token index 和 scVI decoder index 必须来自各自的 vocabulary。
5. 只有正式候选 checkpoint 通过以上检查后，才把 `configs/davf_ko.yaml` / `configs/davf_kd.yaml` 的历史默认路径切换到新目录。切换配置本身另做一次显式变更，避免配置文件提前宣称模型已验收。

## 6. 当前实测结果

| 检查 | 结果 |
|---|---|
| KO/KD cell-baseline pair 重建 | 通过；KO `2560/2524/2522`，KD `163844/22320/21725`；same-batch 控制细胞数组完整且 split-disjoint |
| `ruff` + `compileall` | 通过 |
| DAVF 相关单元测试 | `50 passed` |
| KO 5 epoch GPU smoke | checkpoint 导出/重载通过；test 1-step relative MSE `0.8766`，cosine `0.3313`，sign `0.5963` |
| KD 5 epoch GPU smoke | checkpoint 导出/重载通过；test 1-step relative MSE `0.5832`，cosine `0.6436`，sign `0.6629` |
| KO/KD 50-step smoke | KO relative MSE `0.8878`；KD `0.7438` |
| KO 40 epoch 正式候选 | checkpoint 导出/重载通过；test 1/50-step relative MSE `0.6053/0.6257`；gene sign `0.6749/0.6657`；inconclusive `0/0` |
| KD 40 epoch 正式候选 | checkpoint 导出/重载通过；test 1/50-step relative MSE `0.5133/0.9346`；gene sign `0.7088/0.5785`；inconclusive `0/0` |
| 新 checkpoint DAVF→PerturbGen bridge | KO/KD 均 `pass`；token 与 decoder index 分离 |
| dated formal config → E2E CLI | KO/KD 真实 prepared H5AD 均 `pass`；`5 passed` |

正式候选已经生成并通过工程验收，但还不能把它们称为最终生物学模型，也没有切换历史默认配置。现有 scVI 是在 prepared 全体细胞上训练的，cell split 仍属于 transductive latent evaluation；若要做 donor-held-out 的严格泛化，需要先按 donor 划分 scVI 训练和 DAVF pair 生成流程。

当前 KD 的 50-step gene sign 明显低于 1-step；在下游 DAVF direction gate 中应先使用
`num_steps=1`，把 50-step 作为多步积分稳定性监测。若要求 50-step 作为正式推理，
需要重新设计覆盖整个积分轨迹的训练目标并重新验收。

当前 PerturbGen 的 `local_adaptation_datlinger2021_v10` 六阶段 manifest 均为
`success`，但训练数据是 750-cell Datlinger-Bock 子集，mask/decoder 各 2 个 epoch；
这只能作为真实工程链证据，不能替代正式多 donor normal/disease 训练。正式 E2E
的下一步是提供满足 donor/state 契约的数据，按候选 route 建立独立输出目录，完成
六阶段训练并用其 held-out utility 指标验收。
