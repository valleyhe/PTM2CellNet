# Data refresh and Ridge baseline maintenance workflow

This workflow is the operational companion to
[`data/manifests/datasets.yaml`](../data/manifests/datasets.yaml). It keeps
source provenance and the classical baseline reproducible without checking
restricted raw data into Git.

## 1. Register a source release

1. Confirm the provider release/date, species, license and allowed storage.
2. Place the user-authorized snapshot outside Git or in an approved local data
   directory.
3. Register the local path and SHA-256:

   ```bash
   python scripts/update_data_manifest.py \
     --manifest data/manifests/datasets.yaml \
     --dataset pmads \
     --path data/processed/pmads_combined.csv \
     --format csv
   ```

4. Validate structure and snapshot integrity:

   ```bash
   python scripts/validate_data_manifest.py \
     --manifest data/manifests/datasets.yaml \
     --check-files --verify-hashes
   ```

The validator treats optional/controlled missing files as warnings, but a
registered `required: true` file or a checksum mismatch is an error.

## 2. Run quality checks and regenerate the baseline

The input must contain a real sequence and a target. `sequence` is never
created from a random fallback. For classification:

```bash
python scripts/baseline_pmads_ridge.py \
  --input data/processed/pmads_combined.csv \
  --target label \
  --task classification \
  --group-col protein_accession \
  --seed 42 \
  --manifest data/manifests/datasets.yaml \
  --output-dir outputs/baselines/pmads_ridge/<release-id>
```

For a continuous PMADS effect/intensity target, use `--task regression` and a
numeric target column. The command writes `ridge_model.joblib`, `metrics.json`,
`predictions.csv` and `baseline_manifest.json`. The latter records the data
digest, manifest digest, split policy, feature schema, code revision and
runtime parameters.

## 3. Review and compare

Before accepting a refresh, review:

- required/provenance columns and the canonical position convention (1-based);
- missing/non-standard sequences, invalid PTM positions and duplicate rate;
- group leakage across train/validation/test;
- test metrics against the prior artifact using the same target/task;
- whether a source release, preprocessing version or split seed changed.

Changes in source release or preprocessing version require a new output
directory and a short entry in the experiment log. Do not overwrite a prior
artifact or compare metrics from different split policies as if they were a
single benchmark.

## 4. Evidence boundary

Fixtures are engineering tests only. A green Ridge smoke test proves that the
pipeline is deterministic and operational; it does not prove PMADS coverage,
biological validity or superiority of the cross-scale model. Those claims need
authorized real assets and a separate scientific acceptance review.
