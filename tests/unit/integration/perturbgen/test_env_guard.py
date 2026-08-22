import sys
import pytest

from src.integration.perturbgen.env_guard import (
    PROJECT_ROOT,
    PerturbGenEnvError,
    probe_external_environment,
    validate_repo_roots,
    sha256_path,
)


def test_importing_env_guard_does_not_import_perturbgen():
    assert not any(name == "perturbgen" or name.startswith("perturbgen.") for name in sys.modules)


def test_top_level_integration_lazy_export_never_imports_external_perturbgen():
    import src.integration as integration

    assert integration.PerturbGenRunner.__name__ == "PerturbGenRunner"
    assert not any(name == "perturbgen" or name.startswith("perturbgen.") for name in sys.modules)


def test_validate_repo_roots_uses_real_workspace_defaults():
    roots = validate_repo_roots()
    assert roots.project_root == PROJECT_ROOT
    assert (roots.perturbgen_repo_root / "perturbgen/__main__.py").is_file()


def test_probe_external_environment_reports_missing_assets_without_importing_perturbgen(tmp_path):
    report = probe_external_environment(
        sys.executable,
        project_root=PROJECT_ROOT,
        perturbgen_repo_root=PROJECT_ROOT / "ref/Perturbgen-src",
        dependency_files=["pyproject.toml"],
        asset_paths=[tmp_path / "missing.ckpt"],
    )
    assert report.ok is False
    assert any("required file missing" in issue for issue in report.issues)
    assert report.python_version
    assert not any(name == "perturbgen" or name.startswith("perturbgen.") for name in sys.modules)


def test_validate_repo_roots_rejects_invalid_external_repo(tmp_path):
    fake_root = tmp_path / "bad_repo"
    fake_root.mkdir()
    with pytest.raises(PerturbGenEnvError, match="invalid PerturbGen repo root"):
        validate_repo_roots(project_root=PROJECT_ROOT, perturbgen_repo_root=fake_root)


def test_probe_rejects_wrong_python_and_commit(tmp_path):
    asset = tmp_path / "asset"
    asset.write_text("x", encoding="utf-8")
    report = probe_external_environment(
        sys.executable,
        project_root=PROJECT_ROOT,
        perturbgen_repo_root=PROJECT_ROOT / "ref/Perturbgen-src",
        asset_paths=[asset],
        expected_perturbgen_commit="0" * 40,
    )
    assert any("expected 3.11" in issue for issue in report.issues)
    assert any("commit mismatch" in issue for issue in report.issues)


def test_sha256_path_supports_dataset_directories(tmp_path):
    dataset = tmp_path / "tokenized.dataset"
    (dataset / "data").mkdir(parents=True)
    (dataset / "data" / "part.arrow").write_bytes(b"a")
    first = sha256_path(dataset)
    (dataset / "data" / "part.arrow").write_bytes(b"b")
    assert sha256_path(dataset) != first
