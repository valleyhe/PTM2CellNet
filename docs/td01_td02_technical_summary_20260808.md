# TD-01 / TD-02 技术总结与可复现报告

- **项目**：PTM2CellNet
- **报告日期**：2026-08-08
- **输入依据**：仓库现有代码、归档的 `project_analysis_20260808_pre_cross_scale.md`、本次实际执行的静态检查/测试/基准命令。
- **证据边界**：本报告不把临时 fixture、随机初始化模型或当前工作区中的受控数据文件当作生物学证据；没有下载外部权重、没有调用外部网络服务，也没有声称完成真实数据训练。

## 1. 执行摘要

本轮将分析报告中两个未闭环的高优先级技术债转化为可运行、可审计的实现：

| 技术债 | 本轮交付 | 当前状态 |
|---|---|---|
| TD-01 跨尺度科学模型缺口 | Ankh39/ESM-2/ProtT5 统一编码、显式信号图与可微 CIGNN 桥接、精确随机游走敏感度矩阵、细胞图/基因解码器、PTM 条件化、provenance | 工程契约和离线 smoke 已闭环；真实权重/图和生物学训练仍需授权数据 |
| TD-02 数据清单与 PMADS 基线缺口 | 15 项数据清单、来源/权限/字段/质量/哈希、快照注册与校验 CLI、分组无泄漏 Ridge 基线、artifact manifest 和维护流程 | 清单、基线、测试和维护文档已闭环；受控数据仍按本地快照管理 |

最终回归结果：全仓库 `pytest` 收集 1794 项，**1781 passed、13 skipped、28 warnings**，耗时 375.81 秒；新增范围定向测试 **28 passed、8 warnings**（其中跨尺度模型 16 项、manifest/Ridge 12 项）。`ruff`、`mypy`、`compileall` 和 `git diff --check` 均通过。R-01～R-03 的后续契约修复与逐轮证据见 [`r01_r03_systematic_repair_report_20260808.md`](r01_r03_systematic_repair_report_20260808.md)。

## 2. 任务判断

本次代码任务同时属于：

1. 新功能实现：新增 opt-in 跨尺度模型和 PMADS Ridge 基线。
2. 数据处理/自动化：新增数据清单、快照哈希注册、清单验证和基线 CLI。
3. 测试补充：新增单元、集成和性能 smoke 测试。
4. 性能验证：新增 CPU 工程基准并记录延迟、吞吐、参数量和 RSS 变化。
5. 需求转代码设计：把原分析报告中的 TD-01/TD-02 设计建议固化为输入/输出/错误处理/provenance 契约。

## 3. 运行环境与资源

本报告中的数字来自本次工作区实际命令，而非历史记录。

| 项目 | 实际值 |
|---|---|
| OS | Linux 6.8.0-136-generic, glibc 2.35 |
| Python | 3.12.13, Anaconda, GCC 14.3.0 |
| PyTorch | 2.4.1+cu118；本轮 benchmark 强制 CPU |
| NumPy / pandas | 2.4.3 / 2.3.3 |
| scikit-learn / joblib | 1.8.0 / 1.5.3 |
| PyYAML | 6.0.3 |
| pytest | 9.0.3 |
| ruff / mypy | 0.15.15 / 2.1.0 |
| Lightning | 2.6.5 |
| FastAPI / Pydantic | 0.138.1 / 2.13.4 |
| 网络/外部权重 | 未使用；`MultiPLMEncoder` 默认只允许本地权重，网络加载必须显式设置 `local_files_only=False` |
| 受控原始数据 | 未加入 Git；PMADS 本地快照只记录路径、字节数和 SHA-256 |

### 3.1 资源边界

- 精确敏感度矩阵使用 dense solve，并将节点数限制为 2048；大图必须另行设计稀疏/迭代实现，不能静默分配超大矩阵。
- 本轮 benchmark 使用 `threads=1`、CPU、固定 seed 42、batch 8、信号节点 32、细胞基因 64，仅用于工程回归，不代表生产容量。
- 当前 smoke 模型参数量为 7,094，全部可训练；没有加载数十亿参数的 Ankh39、ESM-2 或 ProtT5 权重。

## 4. TD-01 跨尺度模型实现

### 4.1 数据流

```text
蛋白序列或预计算 pLM embedding
        |
        v
[MultiPLMEncoder]
  Ankh39 + ESM-2 + ProtT5
  显式 residue_alignment / 缺失权重错误
        |
        v
[PTM type + 1-based position conditioning]
        |
        v
[CIGNNSignalBridge]
  显式 PPI/kinase-substrate graph
  S=(1-alpha)(I-alpha G')^-1
  typed edges / edge weights / node mask
        |
        v
[CellGraphCompassHead]
  signal-to-gene map -> gene graph propagation
  gene mask -> delta expression + cell-state logits
        |
        v
输出：预测、logits、概率、敏感度矩阵、protein/signal embedding、provenance
```

### 4.2 关键契约

| 层 | 输入契约 | 输出/失败行为 |
|---|---|---|
| pLM | `ankh39_embeddings`、`esm2_embeddings`、`prott5_embeddings` 可为 `[B,L,D]`；不同 tokenizer 长度必须提供 `residue_alignment`；raw sequence 由 tokenizer 映射到 residue nodes | 融合后的 `[B,L,D_out]`；缺少 required backbone、无法一一对齐 residue 或未提供对齐矩阵时抛出 `CrossScaleContractError` |
| PTM | `ptm_types` 和 `ptm_positions` 必须成对出现，形状 `[B,P]`，位置按 1-based 输入；活动位点越界/未知类型直接失败 | `PTMTokenAdapter` 将 type/position token scatter-add 到蛋白节点；缺字段不会静默丢失 |
| 信号图 | `signal_edge_index`/`edge_index` 必须为 `[2,E]`；权重非负；typed graph 需要 `signal_edge_type` | 可微图传播和 `sensitivity_matrix`；默认不构造全连接生物图 |
| 细胞图 | `cell_edge_index` 和 `signal_gene_map` 默认必需；`gene_mask` 可选 | masked delta-expression、cell-state logits、gene embeddings；缺图/映射抛出契约错误 |
| provenance | 模型版本、数据清单、图版本、seed、pLM 来源、fallback/graph approximation 状态 | 随 forward 结果返回，便于检查实验是否使用了 fallback 或近似图 |

### 4.3 科学解释

`compute_sensitivity_matrix` 实现的是报告要求的

\[
S=(1-\alpha)(I-\alpha G')^{-1}
\]

其中 `G'` 为带自环、按行归一化的 source→destination 图。对当前约定，`S` 的每一行和为 1；它可解释为信号图上的扩散/随机游走影响权重。CIGNN 在初始节点特征和每个 message layer 中使用该矩阵，因此图影响可以反向传播到输入节点和边权。

这提供了“蛋白/位点 → 信号网络 → 基因/细胞状态”的可训练结构，但不自动等同于因果效应。只有在真实扰动设计、正确时间/剂量信息、授权的表达观测和独立验证集存在时，`delta_expression` 才能被解释为经验预测的扰动变化。

### 4.4 兼容性策略

- 现有默认 `PTM2CellNetBase`、`PTM2CellNet` 和 `PTM2CellNetLarge` 构造路径未被替换。
- 新模型通过 `CrossScalePTM2CellNet` / `CrossScaleModel` 和 `configs/cross_scale/smoke.yaml` opt-in。
- pLM 可接收预计算 embedding；本地 HuggingFace 权重加载是独立方法，不在 import 或构造阶段下载。
- `allow_plm_fallback`、`allow_graph_approximation` 默认均为 `false`；打开后 provenance 会明确记录，适合工程消融而不是默认科学结论。

## 5. TD-02 数据清单与 PMADS Ridge 基线

### 5.1 数据库存量清单

`data/manifests/datasets.yaml` 当前包含 15 项，清单 digest 为：

```text
0603113cad374ab6812b9b6e0940b1e4fe29b612df897936dcebd5a3cd600337
```

| 分类 | 条目 |
|---|---|
| 蛋白序列 | UniProtKB |
| PTM 注释 | PhosphoSitePlus、dbPTM、CPLM、PTMAtlas |
| PTM 实验证据 | ProteomeTools / PROSPECT |
| PMADS 监督 | PMADS/CPTAC |
| 单细胞扰动 | scPerturb、Norman/Adamson、Replogle |
| 细胞状态参考 | scGeneScope |
| 信号图 | kinase-substrate、STRING、BioPlex、RegNetwork |

状态计数为：`implemented=4`、`local=1`、`controlled=10`。其中 PMADS 的两个本地快照仍被 `.gitignore` 排除，但已记录：

| 快照 | 字节数 | SHA-256 |
|---|---:|---|
| `data/ptmdb/CPTAC_PTM_intensity.csv` | 8,149,441 | `b64c3fa3a7493b2bedce05a9fa11b983e2ab585f435eb721202079459a927b2c` |
| `data/ptmdb/PTMD_data_in_pmads.txt` | 3,064,389 | `5bb28deb7c559312663a1df58ef6bcce781decab88715b6fa3241b58fd8c7df2` |

清单同时记录 provider、homepage/reference、访问权限、license、格式、loader、canonical schema、1-based position 语义和质量检查。`kinase_substrate` 单独列出 `source_gene`、`target_gene`、`edge_type`，避免把 PPI 边和激酶-底物边混成同一个无类型图。

### 5.2 清单维护流程

1. 数据所有者确认 release、物种、许可和本地存储权限。
2. 快照置于 Git 外或被批准的本地目录。
3. 使用 `scripts/update_data_manifest.py` 记录路径、格式、字节数、日期和 SHA-256；重复更新同一文件只替换对应条目，新增不同文件不会覆盖已有快照。
4. 使用 `scripts/validate_data_manifest.py --check-files --verify-hashes` 检查结构、存在性和哈希漂移。
5. 变更 source release、预处理版本或 split seed 时，写入新的 baseline 输出目录，不覆盖旧 artifact。

具体操作规程见 [`docs/DATA_UPDATE_WORKFLOW.md`](DATA_UPDATE_WORKFLOW.md) 和 [`data/manifests/README.md`](../data/manifests/README.md)。

### 5.3 Ridge baseline 设计

`src/baselines/pmads_ridge.py` 的基线特征保持透明且不依赖深度模型：

- 20 种氨基酸组成、序列长度；
- PTM 总数、位置计数、归一化位置统计；
- PTM type one-hot；
- 显式选择的数值测量列；
- `StandardScaler -> sklearn.linear_model.Ridge`。

默认分类参数为 `alpha=1.0`、`seed=42`、`test_size=0.2`、`validation_size=0.1`。提供 `protein_accession` 时使用两次 seeded `GroupShuffleSplit`，保证蛋白组不跨 train/validation/test；没有 group 列时才回退到确定性的 stratified/shuffle split。所有分区都有不重叠且覆盖全部输入行的防御性断言。

输入缺少真实 `sequence`、target、PTM JSON 不合法或序列为空时直接失败；代码不会生成随机/占位序列。输出包括：

```text
ridge_model.joblib
metrics.json
predictions.csv
baseline_manifest.json
```

manifest 保存输入 digest、数据清单 digest、Git revision、Python/platform、模型类、split 策略、特征 schema、质量 profile 和运行参数。`--demo-data` 会将 artifact 标记为 `model_kind=demo`、`not_for_biological_use=true`。

## 6. 定量验证结果

### 6.1 全量测试

```text
python -m pytest -q
collected 1794 items
1781 passed, 13 skipped, 28 warnings in 375.81s (0:06:15)
```

真实资产目录中的测试按项目 gate 全部 skip；这表示环境没有相应真实资产，不表示真实服务/权重验证通过。

### 6.2 新增范围测试

```text
python -m pytest \
  tests/unit/data/test_data_manifest.py \
  tests/unit/baselines/test_pmads_ridge.py \
  tests/unit/models/test_cross_scale.py \
  tests/integration/test_data_manifest_cli.py \
  tests/integration/test_pmads_ridge_cli.py \
  tests/integration/test_cross_scale_pipeline.py \
  tests/integration/test_cross_scale_benchmark.py -q
28 passed, 8 warnings
```

覆盖内容包括：敏感度矩阵公式/随机性/非法边、CIGNN 梯度、typed edge、三路 pLM 融合、缺失权重和 residue alignment、细胞图 mask/loss、完整 forward-loss-backward-checkpoint、manifest 哈希漂移、CLI 快照注册、多文件不覆盖、Ridge 分组切分和回归/分类分支。

### 6.3 静态质量门

```text
python -m ruff check src scripts tests        -> All checks passed
python -m mypy src                             -> Success: no issues found in 128 source files
python -m compileall -q src scripts tests      -> exit 0
git diff --check                               -> exit 0
```

### 6.4 跨尺度工程 benchmark

参数：`batch=8`、`signal_nodes=32`、`cell_genes=64`、`iterations=5`、`warmup=1`、`seed=42`、`threads=1`、CPU。

| 指标 | 值 |
|---|---:|
| 总参数 / 可训练参数 | 7,094 / 7,094 |
| mean latency | 1.1680 ms |
| p50 latency | 1.1525 ms |
| p95 latency | 1.2544 ms |
| 吞吐 | 6,849.12 samples/s |
| RSS before / after | 888.625 / 888.750 MB |
| RSS delta | 0.125 MB |
| 输出形状 | `delta_expression=[8,64]`, `cell_state_logits=[8,4]`, `sensitivity=[32,32]` |

这些数字是小型内存 fixture 的 CPU forward smoke，不是大图、真实 pLM 或生产服务 benchmark。

### 6.5 PMADS Ridge CLI smoke

本次用临时的 36 行工程 fixture 执行真实 CLI，`--group-col protein_accession`、`seed=42`、`alpha=1.0`、`--demo-data`。分区为 train=21、validation=6、test=9，策略为 `group_shuffle_split`，CLI 返回码为 0，四个 artifact 均生成。该 fixture 的 test accuracy/balanced accuracy/macro-F1/MCC 均为 1.0；由于它是工程 fixture 且被标记 `not_for_biological_use`，这些完美分数不能解释为 PMADS 生物学效果。

### 6.6 Manifest 校验

```text
python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --check-files --verify-hashes
ok=true, dataset_count=15, errors=[], warnings=[]
```

## 7. 异常、根因与处理

| 现象 | 根因 | 处理 |
|---|---|---|
| 分析报告指出 TD-01/TD-02 未实现 | 旧代码只有单尺度预测/启发式 pathway mapper，缺少跨尺度图和可复现 baseline | 新增独立 opt-in 模型、manifest、Ridge 和完整测试链 |
| 不同 pLM tokenizer 长度不一致 | special token/分词轴不能按位置直接相加 | 未提供 `residue_alignment` 时硬失败；提供显式 `[L_backbone,L_target]` 映射后再融合 |
| pLM 权重或 signal graph 缺失 | 真实外部资产不在仓库 | 默认硬失败；fallback/全连接近似只能显式开启并记录 provenance |
| Ridge artifact 初版试图序列化 DataFrame | split 对象直接写 JSON 不可序列化 | manifest 只保存 split metadata 和 indices，DataFrame 只用于运行期 |
| 更新 PMADS 第二个文件会覆盖第一个文件 | 更新脚本初版总是替换 `files[0]` | 按 `path` 定位替换，未找到时追加；新增多文件集成回归测试 |
| 项目 raw data 默认被 `.gitignore` 忽略 | 受控数据不能随代码提交 | 仅白名单 `data/manifests/**`，保留快照在本地并记录哈希 |
| 测试有 28 个 warning | 主要来自 Mamba CUDA AMP、Pydantic V1 API、Starlette/httpx、Lightning worker 建议和已取消功能兼容层 | 当前不阻断；后续可独立升级依赖/迁移 API，未把 warning 伪装成失败 |

## 8. 可复现命令

```bash
# 1) 数据清单
python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --check-files --verify-hashes

# 2) 跨尺度 opt-in 配置/测试
python -m pytest tests/unit/models/test_cross_scale.py \
  tests/integration/test_cross_scale_pipeline.py -q
python scripts/benchmark_cross_scale.py \
  --batch-size 8 --signal-nodes 32 --cell-genes 64 \
  --iterations 10 --warmup 2 --threads 1

# 3) 使用授权 PMADS 快照运行基线
python scripts/baseline_pmads_ridge.py \
  --input data/processed/pmads_combined.csv \
  --target label --task classification \
  --group-col protein_accession --seed 42 --alpha 1.0 \
  --manifest data/manifests/datasets.yaml \
  --output-dir outputs/baselines/pmads_ridge/<release-id>

# 4) 质量门
python -m ruff check src scripts tests
python -m mypy src
python -m compileall -q src scripts tests
python -m pytest -q
```

## 9. 限制与下一步

1. 当前跨尺度模块完成的是科学模型结构、输入输出契约和工程可验证性，不是已在 PMADS/CPTAC/单细胞真实数据上训练完成的模型。
2. Ankh39、ESM-2、ProtT5 的权重版本、tokenizer 版本和许可需要在真实实验前写入数据/模型 manifest；当前 smoke 使用预计算随机 embedding 仅为 shape/梯度验证。
3. signal graph、signal-to-gene map、cell graph 的 release、物种、阈值和 gene-ID 映射需要真实实验负责人确认；尤其不能把不同证据类型的边无标记合并。
4. dense sensitivity solve 适合小图和单元测试；真实大图应实现显式 sparse/iterative sensitivity，并继续对数值稳定性、显存和 batch 规模做 benchmark。
5. Ridge 只作为透明参考线；与深度模型比较时必须使用相同 release、相同 group split、相同 target 语义和独立 test 集。
6. 真实资产测试当前为 skip；下一阶段应在不提交受控数据的前提下，由数据所有者挂载快照后执行 manifest hash、质量审计、baseline 和科学 acceptance review。

## 10. 需求追踪

| 需求 | 代码/文档 | 验证 |
|---|---|---|
| TD-01 pLM 多尺度融合 | `src/models/cross_scale.py`, `configs/cross_scale/smoke.yaml` | pLM fusion、alignment、missing-weight unit tests；pipeline forward/backward |
| TD-01 CIGNN/敏感度 | `src/models/cross_scale.py` | formula、row-sum、gradient、typed-edge tests；benchmark |
| TD-01 细胞图解码 | `src/models/cross_scale.py` | graph/map/mask/loss tests；pipeline checkpoint round-trip |
| TD-02 数据清单 | `data/manifests/datasets.yaml`, `src/data/data_manifest.py` | structural + local file + SHA-256 CLI validation |
| TD-02 快照维护 | `scripts/update_data_manifest.py`, `scripts/validate_data_manifest.py`, `docs/DATA_UPDATE_WORKFLOW.md` | 6 manifest tests，含多文件不覆盖回归 |
| TD-02 Ridge 基线 | `src/baselines/pmads_ridge.py`, `scripts/baseline_pmads_ridge.py` | 分类/回归、缺失输入、组切分、artifact provenance、CLI |
| 项目状态追踪 | `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md` | Phase 18/19/20 规划与验证记录 |
