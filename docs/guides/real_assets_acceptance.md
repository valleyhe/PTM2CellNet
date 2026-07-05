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
