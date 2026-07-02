# Project Analysis Documentation Maintenance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible maintenance workflow that keeps the deep project explanation document synchronized with code and external references.

**Architecture:** The plan adds lightweight doc-check scripts, reference validation tests, and CI-style verification commands. It avoids changing model behavior and focuses on maintaining documentation accuracy as code evolves.

**Tech Stack:** Python, pytest, Markdown, FastAPI/PyTorch repo conventions

---

### Task 1: Add document section integrity checks

**Files:**
- Create: `tests/unit/test_analysis_doc_integrity.py`
- Test: `tests/unit/test_analysis_doc_integrity.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_analysis_doc_contains_required_sections():
    text = Path("docs/plans/2026-03-08-current-codebase-deep-analysis.md").read_text(encoding="utf-8")
    required = [
        "全局架构与端到端数据流",
        "模块级分析",
        "关键算法总览表",
        "同类方案横向比较",
        "参考资源",
    ]
    for key in required:
        assert key in text
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_analysis_doc_integrity.py::test_analysis_doc_contains_required_sections -q`  
Expected: FAIL (new test file absent before implementation)

**Step 3: Write minimal implementation**

Add the test file with required-section assertions only.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_analysis_doc_integrity.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_analysis_doc_integrity.py
git commit -m "test: add integrity checks for analysis documentation sections"
```

### Task 2: Validate key reference links format and presence

**Files:**
- Modify: `tests/unit/test_analysis_doc_integrity.py`
- Test: `tests/unit/test_analysis_doc_integrity.py`

**Step 1: Write the failing test**

```python
import re
from pathlib import Path


def test_analysis_doc_has_sufficient_references():
    text = Path("docs/plans/2026-03-08-current-codebase-deep-analysis.md").read_text(encoding="utf-8")
    links = re.findall(r"https?://[^\s)]+", text)
    assert len(links) >= 10
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_analysis_doc_integrity.py::test_analysis_doc_has_sufficient_references -q`  
Expected: FAIL if link count or regex check does not pass initially

**Step 3: Write minimal implementation**

Adjust test and/or document references to satisfy minimum count and format checks.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_analysis_doc_integrity.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_analysis_doc_integrity.py docs/plans/2026-03-08-current-codebase-deep-analysis.md
git commit -m "docs: enforce reference-link presence checks for analysis document"
```

### Task 3: Add module coverage mapping check in docs

**Files:**
- Modify: `tests/unit/test_analysis_doc_integrity.py`
- Test: `tests/unit/test_analysis_doc_integrity.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_analysis_doc_mentions_all_core_modules():
    text = Path("docs/plans/2026-03-08-current-codebase-deep-analysis.md").read_text(encoding="utf-8")
    modules = ["src/data", "src/models", "src/training", "src/evaluation", "src/utils", "src/api"]
    for module in modules:
        assert module in text
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_analysis_doc_integrity.py::test_analysis_doc_mentions_all_core_modules -q`  
Expected: FAIL if any module string is absent

**Step 3: Write minimal implementation**

Update doc text to explicitly include all module paths if needed.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_analysis_doc_integrity.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_analysis_doc_integrity.py docs/plans/2026-03-08-current-codebase-deep-analysis.md
git commit -m "test: add module mapping assertions for analysis documentation"
```

### Task 4: Add reproducible verification command block

**Files:**
- Modify: `docs/plans/2026-03-08-current-codebase-deep-analysis.md`

**Step 1: Write the failing test**

N/A (documentation task).

**Step 2: Run test to verify it fails**

N/A.

**Step 3: Write minimal implementation**

Append a “Verification Commands” section with exact commands:
- `python -m mypy src`
- `flake8 src tests scripts`
- `python -m pytest -q tests/unit tests/integration --cov=src --cov-report=term-missing --cov-fail-under=80`

**Step 4: Run test to verify it passes**

Run:
- `python -m pytest tests/unit/test_analysis_doc_integrity.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/plans/2026-03-08-current-codebase-deep-analysis.md
git commit -m "docs: add reproducible verification command section"
```

### Task 5: Full quality gate for doc-maintenance changes

**Files:**
- Test: `tests/unit/test_analysis_doc_integrity.py`

**Step 1: Write the failing test**

If quality gate fails, write one minimal failing regression test to pinpoint the issue.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_analysis_doc_integrity.py`  
Expected: FAIL on the targeted issue

**Step 3: Write minimal implementation**

Apply smallest doc/test fix to satisfy gate.

**Step 4: Run test to verify it passes**

Run:
- `python -m mypy src`
- `flake8 src tests scripts`
- `python -m pytest -q tests/unit tests/integration --cov=src --cov-report=term --cov-fail-under=80`

Expected:
- mypy success
- flake8 clean
- tests pass with coverage >= 80

**Step 5: Commit**

```bash
git add tests/unit/test_analysis_doc_integrity.py docs/plans/2026-03-08-current-codebase-deep-analysis.md
git commit -m "test: finalize analysis-doc maintenance quality gate"
```
