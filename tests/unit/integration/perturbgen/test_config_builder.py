import sys
from pathlib import Path

import pytest
from src.integration.perturbgen.config_builder import (
    PerturbGenConfigError,
    build_stage_plans,
    load_pipeline_config,
)
from src.integration.perturbgen.env_guard import PROJECT_ROOT
from scripts.run_perturbgen_pipeline import _build_selected_plans


TEMPLATE = PROJECT_ROOT / "configs/integration/perturbgen.yaml"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _template_env(tmp_path, monkeypatch):
    asset_dir = tmp_path / "assets"
    output_dir = tmp_path / "out"
    values = {
        "PTM2CELLNET_PERTURBGEN_PYTHON": sys.executable,
        "PTM2CELLNET_PERTURBGEN_OUTPUT_ROOT": str(output_dir),
        "PTM2CELLNET_PERTURBGEN_ENCODER_CKPT": str(_touch(asset_dir / "encoder.ckpt")),
        "PTM2CELLNET_PERTURBGEN_TOKEN_DICT": str(_touch(asset_dir / "token.pkl")),
        "PTM2CELLNET_PERTURBGEN_GENE_MAPPING": str(_touch(asset_dir / "mapping.pkl")),
        "PTM2CELLNET_PERTURBGEN_INPUT_H5AD": str(_touch(asset_dir / "input.h5ad")),
        "PTM2CELLNET_PERTURBGEN_DATASET_NAME": "demo",
        "PTM2CELLNET_PERTURBGEN_CELLTYPE_OBS": "cell_type",
        "PTM2CELLNET_PERTURBGEN_STATE_OBS": "state",
        "PTM2CELLNET_PERTURBGEN_DONOR_OBS": "donor",
        "PTM2CELLNET_PERTURBGEN_REFERENCE_STATE": "normal",
        "PTM2CELLNET_PERTURBGEN_TARGET_STATE": "disease",
        "PTM2CELLNET_PERTURBGEN_GENE_MEDIAN": str(_touch(asset_dir / "median.pkl")),
        "PTM2CELLNET_PERTURBGEN_SRC_DATASET": str(_touch(asset_dir / "src.dataset")),
        "PTM2CELLNET_PERTURBGEN_TGT_DATASET_DIR": str(_touch(asset_dir / "tgt.dataset")),
        "PTM2CELLNET_PERTURBGEN_SRC_H5AD": str(_touch(asset_dir / "src.h5ad")),
        "PTM2CELLNET_PERTURBGEN_TGT_H5AD_DIR": str(_touch(asset_dir / "tgt.h5ad")),
        "PTM2CELLNET_PERTURBGEN_MAPPING_DICT": str(_touch(asset_dir / "mapping_dict.pkl")),
        "PTM2CELLNET_PERTURBGEN_MASK_CKPT": str(_touch(asset_dir / "mask.ckpt")),
        "PTM2CELLNET_PERTURBGEN_TOKENID_TO_ROWID": str(_touch(asset_dir / "tokenid_to_rowid.pkl")),
        "PTM2CELLNET_PERTURBGEN_TARGET_GENE": "ENSG000001",
        "PTM2CELLNET_PERTURBGEN_DECODER_CKPT": str(_touch(asset_dir / "decoder.ckpt")),
        "PTM2CELLNET_PERTURBGEN_EMBEDDING_TENSOR_KEY": "model.embed.weight",
        "PTM2CELLNET_PERTURBGEN_EMBEDDING_VOCAB": str(_touch(asset_dir / "vocab.json")),
        "PTM2CELLNET_PERTURBGEN_TOKENISE_OUTPUT": "tokenised_dataset",
        "PTM2CELLNET_PERTURBGEN_MASK_OUTPUT_CKPT": "model/mask.ckpt",
        "PTM2CELLNET_PERTURBGEN_DECODER_OUTPUT_CKPT": "model/decoder.ckpt",
        "PTM2CELLNET_PERTURBGEN_PERTURBATION_SEQUENCE": "src",
        "PTM2CELLNET_PERTURBGEN_PERT_TP": "1",
        "PTM2CELLNET_PERTURBGEN_PERTURB_OUTPUT_H5AD": "results/perturbed.h5ad",
        "PTM2CELLNET_PERTURBGEN_ESTIMATED_OUTPUT_BYTES": "1",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return output_dir


def test_build_stage_plans_from_template_generates_fixed_six_stages(tmp_path, monkeypatch):
    output_dir = _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    plans = build_stage_plans(config, project_root=PROJECT_ROOT)

    assert [plan.name for plan in plans] == [
        "tokenise",
        "train_mask",
        "train_decoder",
        "perturb",
        "export_gene_embeddings",
        "report",
    ]
    perturb = plans[3]
    assert perturb.argv[-2] == "--config"
    assert perturb.generated_files
    assert perturb.generated_files[0].format == "yaml"
    assert perturb.generated_files[0].path.parent.parent == output_dir / "perturb"
    perturb_payload = perturb.generated_files[0].payload
    assert perturb_payload["trainer"]["perturbation_sequence"] == ["src"]
    assert perturb_payload["model"]["ckpt_masking_path"] == "@artifact:train_decoder:checkpoint"
    assert perturb.expected_outputs[0].kind == "h5ad"
    assert perturb.expected_outputs[0].name == "result_h5ad"
    assert perturb.expected_outputs[0].discover_glob == ("*_minference_adata_gENSG000001_s*_tmask.h5ad")
    tokenise = plans[0]
    assert {output.name for output in tokenise.expected_outputs} == {
        "src_dataset",
        "tgt_dataset_folder",
        "src_h5ad",
        "tgt_h5ad_folder",
        "rowid_to_gene_name",
        "tokenid_to_rowid",
    }
    assert all(str(output.path).startswith(str(PROJECT_ROOT / "ref/T_perturb")) for output in tokenise.expected_outputs)
    train_mask = plans[1]
    assert "--src_dataset" in train_mask.argv
    assert "--src-dataset" not in train_mask.argv
    assert train_mask.expected_outputs[0].name == "checkpoint"
    assert train_mask.expected_outputs[0].discover_glob == ("*_train_masking_*-epoch=*.ckpt")
    train_decoder = plans[2]
    assert "@artifact:train_mask:checkpoint" in train_decoder.argv
    assert train_decoder.fingerprint_paths[0] == "@artifact:train_mask:checkpoint"
    export = plans[4]
    assert "--tensor-key" in export.argv
    report = plans[-1]
    assert report.argv == ()
    assert not any(name == "perturbgen" or name.startswith("perturbgen.") for name in sys.modules)


def test_template_does_not_embed_machine_absolute_paths():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "/home/scu/" not in text


def test_cli_both_path_builds_distinct_src_and_tgt_perturb_plans(tmp_path, monkeypatch):
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    plans = _build_selected_plans(config, {"perturb"}, "both")
    assert [plan.name for plan in plans] == ["source_intervention", "within_state"]
    assert plans[0].output_dir != plans[1].output_dir
    assert plans[0].generated_files[0].payload["trainer"]["perturbation_sequence"] == ["src"]
    assert plans[1].generated_files[0].payload["trainer"]["perturbation_sequence"] == ["tgt"]


def test_build_stage_plans_rejects_output_path_escape(tmp_path):
    cfg = {
        "schema_version": 1,
        "repo": {"perturbgen_repo": str(PROJECT_ROOT / "ref/Perturbgen-src")},
        "environment": {"python": sys.executable, "dependency_files": [], "asset_paths": []},
        "pipeline": {
            "perturbgen_commit": "a9a9375",
            "random_seed": 42,
            "output_root": str(tmp_path / "out"),
            "estimated_total_output_bytes": 1,
            "stage_versions": {
                name: 1
                for name in ("tokenise", "train_mask", "train_decoder", "perturb", "export_gene_embeddings", "report")
            },
            "dry_run_estimates": {},
        },
        "stages": {},
    }
    for stage in ("tokenise", "train_mask", "train_decoder", "export_gene_embeddings"):
        cfg["stages"][stage] = {
            "driver": "script",
            "script_root": "project",
            "script_path": str(PROJECT_ROOT / "scripts/export_perturbgen_gene_embeddings.py"),
            "timeout_seconds": 10,
            "uses_gpu": False,
            "expected_outputs": [{"path": "ok.txt", "kind": "file"}],
            "args": {},
        }
    cfg["stages"]["perturb"] = {
        "driver": "perturb_script",
        "timeout_seconds": 10,
        "uses_gpu": False,
        "expected_outputs": [{"path": "ok.txt", "kind": "file"}],
        "perturb_config": {"data": {}, "trainer": {}, "datamodule": {}, "model": {}},
    }
    cfg["stages"]["report"] = {
        "driver": "internal_report",
        "timeout_seconds": 10,
        "uses_gpu": False,
        "expected_outputs": [{"path": "../../escape.json", "kind": "json"}],
    }

    with pytest.raises(PerturbGenConfigError, match="escapes output root"):
        build_stage_plans(cfg, project_root=PROJECT_ROOT)


def test_build_stage_plans_rejects_script_outside_declared_root(tmp_path, monkeypatch):
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    config["stages"]["export_gene_embeddings"]["script_path"] = str(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("", encoding="utf-8")
    with pytest.raises(PerturbGenConfigError, match="escapes declared script_root"):
        build_stage_plans(config, project_root=PROJECT_ROOT)


def test_build_stage_plans_rejects_unsafe_discovery_glob(tmp_path, monkeypatch):
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    config["stages"]["train_mask"]["expected_outputs"][0]["discover_glob"] = "../*.ckpt"
    with pytest.raises(PerturbGenConfigError, match="must stay below"):
        build_stage_plans(config, project_root=PROJECT_ROOT)


def test_tokenise_output_formula_cannot_drift_from_tokenise_args(tmp_path, monkeypatch):
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    config["stages"]["tokenise"]["args"]["n_hvg"] = 3000

    with pytest.raises(PerturbGenConfigError, match="does not match upstream path formula"):
        build_stage_plans(config, project_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset", "../escape", "dataset must be one safe path segment"),
        ("reference_time", "../normal", "time points must be.*safe path segments"),
    ],
)
def test_tokenise_contract_rejects_path_components(tmp_path, monkeypatch, field, value, message):
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    config["stages"]["tokenise"]["args"][field] = value

    with pytest.raises(PerturbGenConfigError, match=message):
        build_stage_plans(config, project_root=PROJECT_ROOT)


def test_perturb_contract_rejects_glob_metacharacters_in_gene(tmp_path, monkeypatch):
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    config["stages"]["perturb"]["perturb_config"]["trainer"]["genes_to_perturb"] = ["ENSG*"]

    with pytest.raises(PerturbGenConfigError, match="target gene must contain only"):
        build_stage_plans(config, project_root=PROJECT_ROOT)
