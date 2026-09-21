# Real-asset acceptance tests (TD-H2, F-01, F-02, F-05)

## Why this exists

The default test suite (`tests/unit`, `tests/integration`, `tests/e2e`) proves
the code is internally correct using **mocks, stubs, and synthetic data**. That
is necessary but not sufficient for production claims. The
`tests/real_assets/` suite closes the gap by running the same code paths
against **real weights, real checkpoints, real hardware, and real network
services**.

| Problem this solves | Without real-assets tests | With real-assets tests |
|---|---|---|
| ESM-3 production load (F-01) | "Mock load works" — but the 2.7GB download, cache hit, and real forward pass are unverified | Forward pass on real `esm3_sm_open_v1` weights is exercised and recorded |
| DAVF E2E finetune (F-02) | Script parses with AST; optimizer is mocked | The script runs as a real subprocess against a real checkpoint + CSV |
| External services (F-05) | UniProt/KEGG/Reactome have fallbacks | The non-fallback path is asserted when the network is genuinely available |
| Cross-scale pLM/graph release | Synthetic embeddings and graph fixtures only | Local pLM completeness plus versioned graph/perturbation snapshots are checked before training |

## DAVF × PerturbGen acceptance boundary

The mainline is `PTM site presence → externally supplied PTM direction proposal →
DAVF direction evidence → donor-level observed expression evidence → direction
gate → PerturbGen invocation`. The PTM classifier only predicts site presence;
`proposed_direction` must come from a user hypothesis or site override. A gate
record is an engineering admission record until its comparison axis is defined.

The observed and DAVF direction fields use different contrasts: observed GSE direction is
donor-level `disease - normal`, while DAVF direction is intervention decode minus
current-context decode. A formal run must record context, intervention, baseline,
objective (disease association, reproduction, or reversal), and source split.
The current gate compares direction strings and does not align or invert these
axes automatically.

Formal acceptance requires all of the following:

- a real normal/disease cohort with raw counts, explicit donor metadata, at least
  three shared donors, and canonical Ensembl IDs;
- an explicit canonical Ensembl mapping for joins; PerturbGen token indices remain
  separate from scVI decoder indices;
- route-specific LatentDAVF/scVI assets with the frozen embedding manifest, exact
  scVI gene order, and a traceable donor split (at least two train and three held
  out donors when claimed);
- data-source and training-split manifests proving that the GSE observed cohort and
  donors are independent of the DAVF training cohort/donors. Current code does not
  automatically validate that isolation;
- a passing direction-gated invocation before any candidate perturb stage;
- both `source_intervention=[src]` (before state transition) and
  `within_state=[tgt]+pert_tps` (within the target state), with at least three
  seeds, matched empirical nulls (at least 99 per required run), extracted
  unperturbed quality, and candidate-level BH-FDR;
- independent replayable manifests for the candidate and null outputs.

The six stages are `tokenise`, `train_mask`, `train_decoder`, `perturb`,
`export_gene_embeddings`, and `report`. `report` only aggregates manifests and
never establishes a biological PASS. The E2E command runs only gate/invocation
by default; `--run-perturbgen` is required to call the external stages. With
`--run-perturbgen`, each route executes the common preparation once under
`<output_root>/<KO|KD>/_prepare/`: `tokenise → train_mask → train_decoder`,
planned by `build_shared_prepare_plans`. The candidate loop sets
`skip_prepare_stages=True` and uses `resolve_prepare_artifact_references` to
resolve shared `@artifact` inputs; a missing reference is a hard failure. Each
candidate then runs only `perturb → export_gene_embeddings → report` in its own
output directory. `--resume` resumes completed stages from manifests in the
same output directory; it is not a separate cross-candidate reuse entry point.
The 2026-09-13 cohort audit has zero formal candidates, so existing smoke,
synthetic, bridge, and checkpoint checks remain engineering evidence only.

## How to run them

These tests are **opt-in**. They never run in default CI.

```bash
# All real-assets tests (requires network + sufficient disk/GPU):
PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 pytest tests/real_assets -v

# Just ESM-3, with a local checkpoint to skip the download:
PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 \
  PTM2CELLNET_ESM3_CHECKPOINT=/path/to/esm3_sm_open_v1.pth \
  pytest tests/real_assets/test_real_esm3.py -v

# Just DAVF E2E:
PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 \
  PTM2CELLNET_DAVF_CHECKPOINT=/path/to/davf.ckpt \
  PTM2CELLNET_DAVF_CSV=/path/to/sites.csv \
  pytest tests/real_assets/test_real_davf_e2e.py -v
```

The cross-scale path has a separate offline release gate. The pLM check only
proves that the local Hugging Face directories are structurally complete; it
does not prove a real forward pass or biological validity:

```bash
python scripts/validate_plm_assets.py \
  --asset-root data/weights/plm \
  --required-backbone ankh39 \
  --required-backbone esm2 \
  --required-backbone prott5

python scripts/validate_data_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --profile cross_scale_training --check-files --verify-hashes
```

The second command is expected to fail in a checkout without the nine
authorized perturbation/signaling snapshots. Do not replace those snapshots
with fixtures and do not interpret a synthetic `train_cross_scale.py` run as a
real-assets or scientific acceptance result.

When the gate env var is unset, every test in `tests/real_assets/` reports
**SKIPPED** with a reason string — so a green default CI run is never mistaken
for "verified against real assets".

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PTM2CELLNET_RUN_REAL_ASSET_TESTS` | unset | Master gate. Must be `1`/`true`/`yes`/`on` to opt in. |
| `PTM2CELLNET_ESM3_CHECKPOINT` | unset | Local path to `esm3_sm_open_v1.pth`. Skips HF download when set. |
| `PTM2CELLNET_ESM3_DEVICE` | auto | `cuda` or `cpu`. |
| `PTM2CELLNET_ESM3_CACHE_DIR` | unset | ESM data cache directory. |
| `PTM2CELLNET_DAVF_CHECKPOINT` | unset | Real DAVF checkpoint for the E2E subprocess test. |
| `PTM2CELLNET_DAVF_CSV` | unset | Real PTM-site CSV for the E2E subprocess test. |
| `PTM2CELLNET_DAVF_EXTRA_ARGS` | unset | Extra CLI args appended to the DAVF script. |
| `PTM2CELLNET_DAVF_TIMEOUT_S` | `1800` | Subprocess timeout for the DAVF E2E test. |

## Evidence capture

Each test writes a JSON record into `outputs/real_assets/<test_name>.json`:

```json
{
  "test": "test_real_esm3_forward",
  "asset": "/path/to/esm3_sm_open_v1.pth",
  "outcome": "pass",
  "duration_s": 18.421,
  "host": "prod-runner-01",
  "platform": "Linux-6.8.0-124-generic-x86_64-with-glibc2.39",
  "timestamp_unix": 1751750400,
  "extra": {
    "model_source": "local_checkpoint",
    "device": "cuda",
    "output_shapes": [[2, 110, 1536]],
    "error": null
  }
}
```

Bundle these JSON files with a release artifact so "we verified ESM-3 against
real weights on `<date>` on `<host>`" is auditable rather than implicit.

## Adding a new real-assets test

1. Create `tests/real_assets/test_real_<feature>.py`.
2. Add `pytestmark = pytest.mark.skipif(not real_assets_enabled(), reason=...)`
   at module scope.
3. Wrap the body in `try/except/finally` and call `record_evidence(...)` in
   the `finally` block so a record is written even on failure.
4. Document any new env vars in the table above.

## Relationship to the default test pyramid

```
                  ┌──────────────────────────┐
                  │ tests/real_assets/       │  opt-in, real weights/network/GPU
                  │ (manual / nightly job)   │
                  └──────────────────────────┘
                ┌──────────────────────────────┐
                │ tests/e2e, tests/integration │  synthetic data, real code paths
                └──────────────────────────────┘
              ┌──────────────────────────────────┐
              │ tests/unit/                      │  mocked deps, fast
              └──────────────────────────────────┘
```

The default CI pyramid must stay green on every commit. The real-assets tier
is a release gate, run by a human or a scheduled job with the right
credentials/hardware, never by default CI.
