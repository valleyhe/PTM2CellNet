# Quality Convergence 90+ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Increase `src` total test coverage from ~84.40% to **90%+** while keeping `mypy=0`, `flake8` clean, and all tests passing.

**Architecture:** Follow a test-only, branch-first strategy. Prioritize high-ROI uncovered branches in `training/*`, then top-up with `evaluation/visualization.py` and `data/features.py`. Use deterministic tiny inputs and strict TDD loops to avoid flaky tests.

**Tech Stack:** Python 3.11, pytest, coverage.py, PyTorch, NumPy, Matplotlib/Seaborn

---

### Task 1: Callbacks branch coverage

**Files:**
- Modify: `tests/unit/test_training.py`
- Test: `tests/unit/test_training.py`

**Step 1: Write the failing test**

```python
def test_model_checkpoint_monitor_missing_logs_and_returns(tmp_path):
    model = SimpleModel()
    trainer = Trainer(model)
    ckpt = ModelCheckpoint(filepath=str(tmp_path / "a.pt"), monitor="val_loss", verbose=0)
    ckpt.on_epoch_end(trainer, 0, {})
    assert ckpt.best_value == float("inf")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_training.py::test_model_checkpoint_monitor_missing_logs_and_returns`  
Expected: FAIL (test missing before implementation)

**Step 3: Write minimal implementation**

Add tests for these branches in `ModelCheckpoint`/`EarlyStopping`:
- missing monitor key
- `save_last=True`
- `save_best_only` improve/no-improve
- `mode="max"`
- early stop trigger after patience

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_training.py -k "checkpoint or early_stopping"`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_training.py
git commit -m "test: add callback branch coverage for checkpoint and early stopping"
```

### Task 2: Trainer guard and scheduler branches

**Files:**
- Modify: `tests/unit/test_training.py`
- Test: `tests/unit/test_training.py`

**Step 1: Write the failing test**

```python
def test_train_epoch_raises_when_not_compiled(sample_data):
    train_loader, _ = sample_data
    trainer = Trainer(SimpleModel())
    with pytest.raises(RuntimeError):
        trainer._train_epoch(train_loader)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_training.py::test_train_epoch_raises_when_not_compiled`  
Expected: FAIL (test missing before implementation)

**Step 3: Write minimal implementation**

Add tests for:
- `_train_epoch` / `_val_epoch` runtime/type errors
- missing `label` / `logits` / `predictions`
- `fit(..., val_loader=None)` path
- callback-triggered early stop path
- `ReduceLROnPlateau` vs non-plateau scheduler path
- `predict` non-dict output raises

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_training.py -k "train_epoch or val_epoch or scheduler or predict"`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_training.py
git commit -m "test: cover trainer guard paths and scheduler branches"
```

### Task 3: Optimizer/scheduler matrix coverage

**Files:**
- Modify: `tests/unit/test_training.py`
- Test: `tests/unit/test_training.py`

**Step 1: Write the failing test**

```python
@pytest.mark.parametrize("name", ["adam", "adamw", "sgd"])
def test_configure_optimizer_supported(name):
    model = SimpleModel()
    cfg = {"training": {"optimizer": name}}
    opt, _ = configure_optimizer(model, cfg)
    assert opt is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_training.py::test_configure_optimizer_supported`  
Expected: FAIL (test missing before implementation)

**Step 3: Write minimal implementation**

Add tests covering:
- optimizer: adam/adamw/sgd/invalid
- scheduler: cosine/plateau/step/invalid
- no-scheduler path

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_training.py -k "configure_optimizer"`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_training.py
git commit -m "test: add optimizer and scheduler matrix coverage"
```

### Task 4: Visualization branch coverage

**Files:**
- Create: `tests/unit/test_visualization.py`
- Test: `tests/unit/test_visualization.py`

**Step 1: Write the failing test**

```python
def test_plot_roc_curve_binary_creates_file(tmp_path):
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    out = tmp_path / "roc.png"
    plot_roc_curve(y_true, y_score, str(out))
    assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_visualization.py::test_plot_roc_curve_binary_creates_file`  
Expected: FAIL (file does not exist yet because test file missing)

**Step 3: Write minimal implementation**

In the new test file, add scenarios for:
- ROC/PR binary + multiclass
- confusion matrix normalize True/False
- training curves with missing optional keys
- feature importance and attention heatmap output files

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_visualization.py`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_visualization.py
git commit -m "test: add visualization branch coverage"
```

### Task 5: Feature extractor branch coverage

**Files:**
- Modify: `tests/unit/test_data.py`
- Test: `tests/unit/test_data.py`

**Step 1: Write the failing test**

```python
@pytest.mark.parametrize("encoding", ["onehot", "kmer", "both"])
def test_extract_sequence_features_encoding_modes(encoding):
    ex = FeatureExtractor(sequence_encoding=encoding)
    arr = ex.extract_sequence_features(["ACDEFGHIKL"])
    assert arr.size > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_data.py::test_extract_sequence_features_encoding_modes`  
Expected: FAIL (test missing before implementation)

**Step 3: Write minimal implementation**

Add tests for:
- encoding mode matrix
- invalid JSON PTM input path
- out-of-range PTM positions path
- `include_ptm_features=False` path
- `combine_features` flattening 1D/2D arrays

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_data.py -k "feature"`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_data.py
git commit -m "test: extend feature extractor branch coverage"
```

### Task 6: Full quality gate and coverage target

**Files:**
- Modify: `tests/unit/test_quality_convergence.py` (optional, if helper assertions needed)
- Test: `tests/unit/`, `tests/integration/`

**Step 1: Write the failing test**

If coverage is still `<90%`, add one focused failing test for the highest uncovered branch from the latest `term-missing` output.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q <new_test_path>::<new_test_name>`  
Expected: FAIL

**Step 3: Write minimal implementation**

Add only the minimal assertions/input setup needed to hit the missing branch.

**Step 4: Run test to verify it passes**

Run:
- `python -m mypy src`
- `flake8 src tests scripts`
- `python -m pytest -q tests/unit tests/integration --cov=src --cov-report=term-missing --cov-fail-under=80`

Expected:
- mypy: `Success: no issues found`
- flake8: no output
- pytest: all pass and total coverage `>= 90%`

**Step 5: Commit**

```bash
git add tests/unit tests/integration
git commit -m "test: reach 90+ coverage with targeted branch tests"
```

### Task 7: Verification snapshot and handoff notes

**Files:**
- Modify: `docs/plans/2026-03-08-quality-convergence-90-design.md` (append implementation result note)

**Step 1: Write the failing test**

N/A (documentation task). Add a result appendix with before/after metrics.

**Step 2: Run test to verify it fails**

N/A

**Step 3: Write minimal implementation**

Append:
- Before coverage
- After coverage
- Test counts
- Known residual gaps (if any)

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit tests/integration --cov=src --cov-report=term --cov-fail-under=80`  
Expected: PASS and reported coverage `>= 90%`

**Step 5: Commit**

```bash
git add docs/plans/2026-03-08-quality-convergence-90-design.md
git commit -m "docs: record quality convergence execution results"
```
