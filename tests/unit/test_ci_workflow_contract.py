"""Static contracts for CI dependency and offline-network policy."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pull_request_ci_has_an_analysis_job_with_anndata_dependency() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "  analysis:" in workflow
    assert "pip install -r requirements-analysis.txt" in workflow
    # U-15（2026-08-24）：覆盖从 3 个点名文件扩展为整个 perturbgen 单测目录，
    # 使 test_data_prep.py 的 anndata 依赖测试不再依赖主 job 的静默 skip。
    assert "tests/unit/integration/perturbgen/" in workflow
    assert "tests/integration/test_perturbgen_pipeline_mocked.py" in workflow


def test_ci_has_dependency_consistency_job() -> None:
    """TD-N-32/34（2026-08-24）：pip check 与 lock 一致性门禁不得被移除。"""
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "  dependencies:" in workflow
    assert "pip install -e ." in workflow
    assert "run: pip check" in workflow
    assert "scripts/check_requirements_consistency.py" in workflow


def test_ci_workflows_make_huggingface_offline_mode_explicit() -> None:
    ci_workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    real_assets_workflow = (PROJECT_ROOT / ".github/workflows/perturbgen-real-assets.yml").read_text(encoding="utf-8")

    for workflow in (ci_workflow, real_assets_workflow):
        assert 'HF_HUB_OFFLINE: "1"' in workflow
        assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
        assert 'HF_DATASETS_OFFLINE: "1"' in workflow
