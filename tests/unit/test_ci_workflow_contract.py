"""Static contracts for CI dependency and offline-network policy."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pull_request_ci_has_an_analysis_job_with_anndata_dependency() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "  analysis:" in workflow
    assert "pip install -r requirements-analysis.txt" in workflow
    assert "tests/unit/integration/perturbgen/test_data_prep.py" in workflow


def test_ci_workflows_make_huggingface_offline_mode_explicit() -> None:
    ci_workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    real_assets_workflow = (
        PROJECT_ROOT / ".github/workflows/perturbgen-real-assets.yml"
    ).read_text(encoding="utf-8")

    for workflow in (ci_workflow, real_assets_workflow):
        assert 'HF_HUB_OFFLINE: "1"' in workflow
        assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
        assert 'HF_DATASETS_OFFLINE: "1"' in workflow
