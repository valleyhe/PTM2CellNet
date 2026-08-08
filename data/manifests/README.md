# Data manifest and update policy

`datasets.yaml` is the authoritative inventory for the data required by the
PTM2CellNet scientific design. It records source, access/licence status,
expected file formats, canonical fields and quality checks. It does not contain
raw data and a `null` path is intentional for controlled or unavailable assets.

Validate the checked-in inventory:

```bash
python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml
```

When a data owner supplies a local snapshot, update one entry and record its
digest (the command does not upload or download anything):

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

Before a baseline refresh, review the provider release/licence, run the quality
checks, regenerate the fixed split with the same seed (or record a deliberate
seed change), and compare the new `baseline_manifest.json` against the previous
artifact. Never replace a missing controlled asset with synthetic data and do
not commit restricted files.
