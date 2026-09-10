# GSE normal/disease 数据接入、DAVF 方向验证与训练计划

**日期**：2026-09-04  
**范围**：GSE214695 健康/ Crohn 病例原始 10x 表达矩阵接入当前 DAVF × PerturbGen 工程

## 1. 先固定科学边界

GSE214695 的 GEO 设计包含 6 个健康对照、6 个 Crohn 病例和 6 个 UC 病例；本轮只下载健康对照与 Crohn 病例，保持二分类 `normal`/`disease`。数据是独立 donor 的观察性 scRNA-seq，不是同一 donor 的配对干预实验。

因此本数据只能承担两件事：

1. 提供独立的 normal/disease 表达方向证据，检验 DAVF 对候选基因方向的预测是否与真实疾病方向一致；
2. 建立一个带 `latent_dim=64`、固定 4018 个 Ensembl 基因轴的 GSE 专用 scVI 坐标资产，用于该数据集内部的状态分析。

它不能承担以下任务：

- 不能直接训练出有因果含义的 KO/KD DAVF checkpoint，因为没有基因级 KO/KD 标签；
- 不能进入当前要求“normal/disease 至少 3 个共享 donor”的 PerturbGen 正式配对流程；
- 不能把 GSE 专用 scVI latent 与 `checkpoints/scvi/davf_ko_dixit` 或
  `checkpoints/scvi/davf_kd_nadig` 的 latent 混用。

KO/KD DAVF 的正式训练仍使用现有真实 Perturb-seq 数据和各自独立的 scVI 坐标；GSE 方向结果通过 canonical Ensembl ID 与 DAVF 结果合并。

## 2. 数据来源与落盘约定

- GEO 系列页：[GSE214695](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214695)
- 每个样本的原始 10x `matrix.mtx.gz`、`features.tsv.gz`、`barcodes.tsv.gz` 从对应 GEO sample supplementary file 下载。
- 细胞注释文件：`GSE214695_cell_annotation.csv.gz`。
- 本轮落盘目录：`data/raw/gse214695/`。
- 选择的样本：`HC1–HC6` 和 `CD1–CD6`，共 12 个样本、两组各 6 个 donor。

原始文件不进入 Git；本机快照及 SHA-256 已登记到 `data/manifests/datasets.yaml`，处理后 H5AD 的 `uns` 还保存了样本、基因轴和冲突注释记录。

## 3. 代码修改计划

| 文件 | 修改 | 完成判据 |
|---|---|---|
| `src/data/gse_normal_disease.py` | 新增通用 GSE 10x 读取、只保留已注释 barcode、冲突注释排除、Ensembl/asset 交集选 4018 基因、raw counts H5AD 生成、donor-level normal/disease 方向汇总 | 能拒绝缺文件、维度不一致、重复 Ensembl、缺注释 barcode、零 library size；冲突标签不猜测而记录并排除；输出包含 `layers["counts"]`、`state`、`donor`、`cell_type` |
| `scripts/prepare_gse_normal_disease.py` | 新增 GSE 数据准备 CLI | 使用实际 12 个样本生成 `data/processed/gse214695_hc_cd/prepared.h5ad` 和 manifest |
| `scripts/summarize_gse_directions.py` | 新增独立方向证据 CLI | 以 donor 为统计单位，输出每个 cell type/gene 的 `log2fc`、p 值、BH-FDR、donor consistency |
| `scripts/train_gse_scvi.py` | 新增 GSE 专用 scVI 训练 CLI | 显式使用 `layer="counts"`、`batch_key="davf_batch"`、`n_latent=64`，保存后通过 `ScVIAdapter` 重载校验 |
| `src/models/scvi_adapter.py` | 修复 scvi-tools 1.4.x 对“仅协变量、全零稀疏 AnnData”误判为 minified 的连接问题；只对 `SCVI.load` 使用保留数值语义的私有副本 | 零表达 decoder context 可加载；真实 minified AnnData 仍交给 scvi-tools 严格报错 |
| `data/manifests/datasets.yaml` | 登记 GSE214695 的来源、输入格式和处理入口，但不把它加入正式 PerturbGen required profile | manifest 只描述来源，不宣称已满足共享 donor 契约 |
| `scripts/build_davf_latent_pairs.py` | 修复 Norman KO pair 元数据，并让 control pool 在 train/val/test 间 disjoint | `load_latent_davf_pairs(..., expected_intervention_type="KO")` 可接受新生成文件；control 只能在同一 split 内复用 |
| `tests/unit/data/test_gse_normal_disease.py` | 覆盖 GSE 读取、未注释 barcode、冲突标签、基因选择、metadata、方向统计和拒绝路径 | 正常、缺注释、冲突注释、非整数 raw counts 均有回归测试 |
| `tests/unit/scripts/test_prepare_gse_normal_disease.py` | 覆盖 CLI 参数和真实准备函数调用 | 小型 10x fixture 可离线生成 H5AD |
| `tests/unit/scripts/test_train_gse_scvi.py` | 只测试参数/契约校验，不在普通单测启动 scVI 训练 | 缺 schema、非 4018 genes、缺 counts 时 fail-fast |
| `tests/unit/test_scvi_adapter.py` | 覆盖全零稀疏 decoder context 的语义保持处理 | 输入仍为全零且原对象不被修改，私有加载副本可通过 scVI 的载入判定 |
| `docs/guides/gse_normal_disease_davf_plan_20260904.md` | 固化边界、命令和验收口径 | 文档命令与实际入口一致 |

不修改 `src/integration/perturbgen/data_prep.py` 的共享 donor 要求；该要求是正式 PerturbGen 配对数据的科学契约，不应为 GSE214695 放宽。

## 4. 执行顺序

1. 下载并检查 12 个样本的原始 10x 三件套和细胞注释。
2. 运行 GSE 准备 CLI，确认最终 H5AD 是 cell × 4018 gene，`layers["counts"]` 为整数型非负 raw counts；GEO 未注释 barcode 不进入结果。
3. 运行离线单元测试、`compileall`、Ruff 和定向集成测试。
4. 在本机 GPU 上以短 epoch 训练 GSE 专用 scVI smoke checkpoint；确认 `ScVIAdapter` 能重载、输出 `[N, 64]` latent，且 gene order 与 H5AD 完全一致。
5. 运行 donor-level normal/disease 方向汇总，生成独立证据 CSV/JSON。
6. 重新生成 Norman KO latent pair 并用严格 loader 校验；必要时再按完整 epoch 训练正式当前 `LatentDAVF` checkpoint。
7. 验证真实 scVI decoder context、当前 DAVF checkpoint 和 PerturbGen token/decoder 双索引路径；不把全零 context 的载入修复扩展成数据伪造。
8. 只有上述检查通过后，才把 GSE 结果作为 DAVF 方向 gate 的 observed evidence；不把它写入 KO/KD DAVF checkpoint 的训练 NPZ。

## 5. 处理与训练命令

准备数据（asset 只用于保证这 4018 个 gene 同时可作为 DAVF 输入 token；不把 token index 当作 scVI 列号）：

```bash
python scripts/prepare_gse_normal_disease.py \
  --raw-dir data/raw/gse214695 \
  --annotation data/raw/gse214695/GSE214695_cell_annotation.csv.gz \
  --normal-samples HC1,HC2,HC3,HC4,HC5,HC6 \
  --disease-samples CD1,CD2,CD3,CD4,CD5,CD6 \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output data/processed/gse214695_hc_cd/prepared.h5ad
```

GSE 专用 scVI 训练（正式长训练建议从 40–100 epochs 开始；本轮先执行短 smoke）：

```bash
python scripts/train_gse_scvi.py \
  --input data/processed/gse214695_hc_cd/prepared.h5ad \
  --output checkpoints/scvi/gse214695_hc_cd \
  --max-epochs 2 --batch-size 1024 \
  --seed 42 --accelerator gpu
```

方向证据汇总：

```bash
python scripts/summarize_gse_directions.py \
  --input data/processed/gse214695_hc_cd/prepared.h5ad \
  --output outputs/gse214695_hc_cd/direction_evidence.csv \
  --min-donors 3
```

Norman KO pair 修复版与 DAVF smoke：

```bash
python scripts/build_davf_latent_pairs.py \
  --norman-dir data/raw/norman_adamson/GSE133344 \
  --scvi-model checkpoints/scvi/ibd_norman_model \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output-dir data/processed/davf_latent/norman_gse133344_v2 \
  --device cpu

python scripts/train_latent_davf.py \
  --train-data data/processed/davf_latent/norman_gse133344_v2/train.npz \
  --val-data data/processed/davf_latent/norman_gse133344_v2/val.npz \
  --scvi-model checkpoints/scvi/ibd_norman_model \
  --intervention-type KO \
  --embedding-asset outputs/perturbgen/embedding_asset_20260822 \
  --output checkpoints/davf/latent_davf_perturbgen_4018_v2 \
  --epochs 100 --device cuda
```

## 6. 训练与验收口径

| 产物 | 可接受结论 |
|---|---|
| `prepared.h5ad` | 数据入口通过；有 raw counts、Ensembl、donor/state/cell_type |
| GSE scVI checkpoint | GSE 专用状态坐标可用；本轮 GPU smoke 已验证；不是 KO/KD DAVF checkpoint |
| direction evidence CSV | 可作为独立 observed direction，按 cell type、gene、donor 复核 |
| Norman KO pair/checkpoint | v2 pair 已通过 strict loader；1 epoch 当前 LatentDAVF smoke checkpoint 已通过 checkpoint contract；这不是完整训练质量结论 |
| 现有 KO/KD DAVF checkpoint | 不因 GSE 下载而自动改变；继续绑定各自 scVI model 和 embedding asset |
| PerturbGen 正式双路径 | GSE214695 本轮不通过共享 donor gate；仍需真正配对 normal/disease donor 队列 |

## 7. 失败策略

所有数据契约错误直接抛出并停止：不填充缺失 donor、不猜 cell type、不把 `.X` 伪装成 raw counts、不用 token index 代替 scVI gene index、不把 observational delta 改名为 KO/KD 标签。

## 8. 本轮实际执行结果

| 检查 | 结果 |
|---|---|
| GSE 原始压缩文件完整性 | 36/36 个样本矩阵文件 gzip 全量读取通过；另有一个未完整下载的 `GSE214695_RAW.tar.partial`，未参与处理 |
| GSE H5AD | `31,800 × 4,018`，含 raw counts、12 个 donor、normal/disease 标签；2 个 CD3 冲突注释 barcode 明确排除 |
| GSE scVI GPU smoke | Tesla P40 上 2 epochs 通过；重载并编码 `[8, 64]` latent 通过 |
| Norman KO latent pair | v2 train/val/test = `46,128/5,349/7,006`；strict KO metadata 与 split-disjoint control pool 通过 |
| Norman DAVF GPU smoke | Tesla P40 上 1 epoch 通过；checkpoint contract 校验通过；不代表正式训练质量 |
| scVI/DAVF/PerturbGen 真实桥接 | 真实集成与 real-assets 测试 `5 passed`；token index 与 scVI decoder index 分离校验通过 |
| 静态与单元测试 | 变更 Python 文件 Ruff/compileall 通过；变更相关定向单元测试 `69 passed, 2 skipped`；完整 `tests/unit` 为 `2205 passed, 6 skipped` |

上述 smoke 产物只证明数据契约、模型维度和调用链可运行；正式模型仍需按完整训练计划重新训练并以独立 donor/perturbation holdout 评估。
