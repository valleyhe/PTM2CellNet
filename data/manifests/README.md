# Data manifest and update policy

`datasets.yaml` is the authoritative inventory for the data required by the
PTM2CellNet scientific design. It records source, access/licence status,
expected file formats, canonical fields and quality checks. It does not contain
raw data and a `null` path is intentional for controlled or unavailable assets.

For the active DAVF × PerturbGen acceptance track (reviewed 2026-09-13), a
manifest entry is only a source registration. It is not a Gate-0 pass. A formal
cohort must separately prove real `normal/disease` raw counts, canonical Ensembl
IDs, explicit donor identity, at least 3 shared donors, scVI gene order, frozen
embedding asset and train-only/held-out donor provenance. The observed
disease−normal contrast is direction evidence, not a KO/KD effect ground truth. The
current formal cohort is 0; PerturbGen token indices and scVI decoder indices are
different spaces and must be aligned through canonical Ensembl IDs and a frozen
manifest, never substituted for one another.

Active execution references are [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md),
the [central dual-path plan](../../docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md),
and [`task_plan.md`](../../task_plan.md).

Validate the checked-in inventory:

```bash
python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml
```

When a data owner supplies a local snapshot, update one entry and record its
digest (the command does not upload or download anything). Do not use an
unspecified or latest file to satisfy a manifest entry:

```bash
python scripts/update_data_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --dataset pmads \
  --path data/processed/pmads_combined.csv \
  --format csv
python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --check-files --verify-hashes
```

发布或研究运行必须显式激活对应 profile；缺少 profile 要求的数据快照、文件或哈希
会 fail-fast：

```bash
python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --profile standard_training --check-files --verify-hashes
python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --profile cross_scale_training --check-files --verify-hashes
```

当前 `standard_training` 的 PMADS 本地快照已通过；`cross_scale_training` 和
`cross_scale_inference` 所需受控图/扰动数据尚未交付，因此预期返回非零状态。

PerturbGen/IBD 队列应在独立 entry 中登记 source、状态、donor、raw-count、Ensembl
和允许用途；四队列规划、smoke 或 synthetic 快照不能替代正式 normal/disease
cohort。候选的 `context`、`intervention`、比较基准、研究目标及 observed/DAVF/
utility 来源和 donor 划分应进入运行 manifest。

Before a baseline refresh, review the provider release/licence, run the quality
checks, regenerate the fixed split with the same seed (or record a deliberate
seed change), and compare the new `baseline_manifest.json` against the previous
artifact. Never replace a missing controlled asset with synthetic data and do
not commit restricted files.
