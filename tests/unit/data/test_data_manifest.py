"""Tests for the versioned data inventory contract."""

from pathlib import Path

import pytest
import yaml

from src.data.data_manifest import (
    DataManifestError,
    get_dataset,
    load_manifest,
    manifest_digest,
    sha256_file,
    validate_manifest,
)


MANIFEST = Path("data/manifests/datasets.yaml")


def _valid_manifest(*, files=None):
    dataset = {
        "id": "fixture",
        "name": "Fixture",
        "category": "test",
        "status": "local",
        "source": {"provider": "tests", "access": "local", "license": "internal"},
        "formats": ["csv"],
        "canonical_schema": {"required_columns": ["value"]},
        "quality_requirements": {"checks": ["non_empty"]},
    }
    if files is not None:
        dataset["files"] = files
    return {"manifest_version": "1.0.0", "project": "test", "datasets": [dataset]}


def test_project_manifest_loads_and_covers_required_sources():
    manifest = load_manifest(MANIFEST)
    ids = {entry["id"] for entry in manifest["datasets"]}
    assert {
        "pmads",
        "ptmatlas",
        "scperturb",
        "kinase_substrate",
        "string",
        "regnetwork",
    }.issubset(ids)
    assert len(ids) == len(manifest["datasets"])


def test_duplicate_dataset_id_is_rejected():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    manifest["datasets"].append(dict(manifest["datasets"][0]))
    report = validate_manifest(manifest)
    assert report["ok"] is False
    assert any("duplicate dataset id" in error for error in report["errors"])


def test_required_snapshot_and_hash_drift_are_actionable(tmp_path):
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = {
        "manifest_version": "1.0.0",
        "project": "test",
        "datasets": [
            {
                "id": "fixture",
                "name": "fixture",
                "category": "test",
                "status": "local",
                "source": {"provider": "test", "access": "local", "license": "internal"},
                "formats": ["csv"],
                "files": [{"path": snapshot.name, "required": True, "sha256": "0" * 64}],
                "canonical_schema": {"required_columns": ["a"]},
                "quality_requirements": {"checks": ["non_empty"]},
            }
        ],
    }
    report = validate_manifest(
        manifest,
        root_dir=tmp_path,
        check_files=True,
        verify_hashes=True,
    )
    assert report["ok"] is False
    assert any("SHA-256 mismatch" in error for error in report["errors"])
    assert sha256_file(snapshot) == report["dataset_reports"][0]["file_checks"][0]["sha256_actual"]


def test_load_manifest_rejects_malformed_yaml_contract(tmp_path):
    malformed = tmp_path / "bad.yaml"
    malformed.write_text("datasets: []\n", encoding="utf-8")
    try:
        load_manifest(malformed)
    except DataManifestError as exc:
        assert "missing required keys" in str(exc)
    else:  # pragma: no cover - protects the contract if the validator regresses
        raise AssertionError("malformed manifest unexpectedly loaded")


def test_manifest_digest_is_stable_across_mapping_order():
    first = {"project": "测试", "datasets": [], "manifest_version": "1"}
    second = {"manifest_version": "1", "datasets": [], "project": "测试"}

    assert manifest_digest(first) == manifest_digest(second)


def test_sha256_file_rejects_missing_path_and_supports_small_chunks(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abcdef")

    assert sha256_file(payload, chunk_size=2) == sha256_file(payload)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sha256_file(tmp_path / "missing.bin")


def test_validate_manifest_reports_schema_type_errors():
    manifest = {
        "manifest_version": "1",
        "project": "test",
        "datasets": [
            "not-a-mapping",
            {
                "id": "",
                "name": "broken",
                "category": "test",
                "status": "unknown",
                "source": [],
                "formats": "csv",
                "canonical_schema": {},
                "quality_requirements": {},
            },
        ],
    }

    report = validate_manifest(manifest)

    assert report["ok"] is False
    messages = "\n".join(report["errors"])
    assert "datasets[0] must be a mapping" in messages
    assert ".id must be a non-empty string" in messages
    assert ".status=" in messages
    assert ".source must be a mapping" in messages
    assert ".formats must be a non-empty list" in messages
    assert ".canonical_schema.required_columns" in messages
    assert ".quality_requirements.checks" in messages


@pytest.mark.parametrize("datasets", [None, {}, "fixture"])
def test_validate_manifest_requires_dataset_list(datasets):
    report = validate_manifest({"manifest_version": "1", "project": "test", "datasets": datasets})

    assert report["ok"] is False
    assert "manifest.datasets must be a list" in report["errors"]


def test_validate_manifest_reports_missing_root_and_dataset_keys():
    report = validate_manifest({"datasets": [{}]})

    assert report["ok"] is False
    assert any("manifest missing required keys" in error for error in report["errors"])
    assert any("datasets[0] missing required keys" in error for error in report["errors"])


def test_file_registration_statuses_without_filesystem_checks(tmp_path):
    manifest = _valid_manifest(
        files=[
            {"path": "registered.csv", "required": True},
            {"path": None, "required": False},
            {"path": 42, "required": False},
        ]
    )

    report = validate_manifest(manifest, root_dir=tmp_path)

    statuses = [item["status"] for item in report["dataset_reports"][0]["file_checks"]]
    assert statuses == ["registered", "not_available", "invalid_path"]
    assert report["ok"] is False
    assert any("path must be a string or null" in error for error in report["errors"])


def test_file_checks_distinguish_required_optional_and_missing_path(tmp_path):
    manifest = _valid_manifest(
        files=[
            {"path": "required.csv", "required": True},
            {"path": "optional.csv", "required": False},
            {"path": None, "required": True},
        ]
    )

    report = validate_manifest(manifest, root_dir=tmp_path, check_files=True)

    statuses = [item["status"] for item in report["dataset_reports"][0]["file_checks"]]
    assert statuses == ["missing", "missing", "missing_path"]
    assert any("required.csv" in error for error in report["errors"])
    assert any("optional.csv" in warning for warning in report["warnings"])
    assert any("required but has no path" in error for error in report["errors"])


def test_hash_validation_covers_match_missing_hash_warning_and_strict_mode(tmp_path):
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("value\n1\n", encoding="utf-8")
    actual_hash = sha256_file(snapshot)

    matching = validate_manifest(
        _valid_manifest(files={"path": str(snapshot), "required": True, "sha256": actual_hash.upper()}),
        check_files=True,
        verify_hashes=True,
    )
    assert matching["ok"] is True
    assert matching["dataset_reports"][0]["file_checks"][0]["status"] == "present"

    missing_hash = validate_manifest(
        _valid_manifest(files=[{"path": str(snapshot), "required": True}]),
        check_files=True,
        verify_hashes=True,
    )
    assert missing_hash["dataset_reports"][0]["file_checks"][0]["status"] == "missing_hash"

    warning = validate_manifest(
        _valid_manifest(files=[{"path": str(snapshot), "sha256": "0" * 64}]),
        check_files=True,
    )
    assert warning["ok"] is True
    assert any("differs" in item for item in warning["warnings"])

    strict = validate_manifest(
        _valid_manifest(files=[{"path": str(snapshot), "sha256": "0" * 64}]),
        check_files=True,
        strict_warnings=True,
    )
    assert strict["ok"] is False
    assert any("warning promoted to error" in item for item in strict["errors"])


def test_load_manifest_handles_missing_invalid_yaml_and_non_mapping(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_manifest(tmp_path / "missing.yaml")

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("datasets: [", encoding="utf-8")
    with pytest.raises(DataManifestError, match="Invalid YAML"):
        load_manifest(invalid_yaml)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("- item\n", encoding="utf-8")
    with pytest.raises(DataManifestError, match="root must be a mapping"):
        load_manifest(scalar)


def test_get_dataset_returns_copy_and_rejects_unknown_id():
    manifest = _valid_manifest()

    dataset = get_dataset(manifest, "fixture")
    dataset["name"] = "changed"

    assert manifest["datasets"][0]["name"] == "Fixture"
    with pytest.raises(KeyError, match="missing"):
        get_dataset(manifest, "missing")


class TestProfileActivation:
    """P1-04: manifest profile 激活校验"""

    def _profile_manifest(self, *, files=None, profile_required=("pmads",)):
        dataset = {
            "id": "pmads",
            "name": "PMADS",
            "category": "ptm_supervision",
            "status": "local",
            "source": {"provider": "tests", "access": "local", "license": "internal"},
            "formats": ["csv"],
            "canonical_schema": {"required_columns": ["sequence"]},
            "quality_requirements": {"checks": ["non_empty"]},
        }
        if files is not None:
            dataset["files"] = files
        return {
            "manifest_version": "1.0.0",
            "project": "test",
            "datasets": [dataset],
            "profiles": {
                "standard_training": {
                    "description": "test",
                    "required_datasets": list(profile_required),
                    "optional_datasets": [],
                }
            },
        }

    def test_profile_required_dataset_missing_path_fails_without_check_files(self):
        """profile required dataset 无本地路径时必须失败（即使未指定 --check-files）"""
        manifest = self._profile_manifest(files=[{"path": None, "required": False}])
        report = validate_manifest(manifest, profile="standard_training")
        assert report["ok"] is False
        assert any("required by profile" in e for e in report["errors"])

    def test_profile_required_dataset_with_path_passes(self, tmp_path):
        """profile required dataset 有本地路径且文件存在时通过"""
        import hashlib

        snapshot = tmp_path / "snap.csv"
        snapshot.write_text("sequence\nACDE\n", encoding="utf-8")
        real_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        manifest = self._profile_manifest(files=[{"path": snapshot.name, "required": False, "sha256": real_hash}])
        report = validate_manifest(
            manifest,
            root_dir=tmp_path,
            check_files=True,
            verify_hashes=True,
            profile="standard_training",
        )
        assert report["ok"] is True
        assert report["profile_required_datasets"] == ["pmads"]

    def test_profile_hash_drift_fails(self, tmp_path):
        """profile required dataset 哈希漂移时必须失败"""
        snapshot = tmp_path / "snap.csv"
        snapshot.write_text("sequence\nACDE\n", encoding="utf-8")
        manifest = self._profile_manifest(files=[{"path": snapshot.name, "required": False, "sha256": "1" * 64}])
        report = validate_manifest(
            manifest,
            root_dir=tmp_path,
            check_files=True,
            verify_hashes=True,
            profile="standard_training",
        )
        assert report["ok"] is False
        assert any("SHA-256 mismatch" in e for e in report["errors"])

    def test_unknown_profile_is_error(self):
        """未知 profile 名必须显式失败（避免无声空校验）"""
        report = validate_manifest(self._profile_manifest(), profile="nope")
        assert report["ok"] is False
        assert any("is not declared" in e for e in report["errors"])

    def test_profile_reference_to_unknown_dataset_is_error(self):
        """profile 引用不存在的 dataset id 必须失败"""
        manifest = self._profile_manifest(profile_required=("missing_dataset",))
        report = validate_manifest(manifest, profile="standard_training")
        assert report["ok"] is False
        assert any("unknown dataset ids" in e for e in report["errors"])

    def test_profile_required_optional_overlap_is_error(self):
        """同一 dataset 同时出现在 required 与 optional 必须失败"""
        manifest = self._profile_manifest()
        manifest["profiles"]["standard_training"]["optional_datasets"] = ["pmads"]
        report = validate_manifest(manifest, profile="standard_training")
        assert report["ok"] is False
        assert any("required and optional" in e for e in report["errors"])

    def test_project_manifest_profiles_declared(self):
        """项目 manifest 必须声明文档阶段 A 的三个研究 profile"""
        manifest = load_manifest(MANIFEST)
        profiles = manifest.get("profiles", {})
        assert {"standard_training", "cross_scale_training", "cross_scale_inference"}.issubset(profiles.keys())
        assert "pmads" in profiles["standard_training"]["required_datasets"]

    def test_profile_unset_keeps_legacy_behavior(self):
        """未指定 profile 时行为不变（无路径不报错）"""
        manifest = self._profile_manifest(files=[{"path": None, "required": False}])
        report = validate_manifest(manifest)
        assert report["ok"] is True
        assert "profile" not in report
