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
LOCAL_ADAPTATION_TEMPLATE = PROJECT_ROOT / "configs/integration/perturbgen_local_adaptation.yaml"


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


def test_local_adaptation_config_uses_upstream_cli_and_runtime_dimensions(monkeypatch):
    monkeypatch.setenv(
        "PTM2CELLNET_PERTURBGEN_PYTHON",
        "/home/scu/anaconda3/envs/perturbgen/bin/python",
    )
    local_output = PROJECT_ROOT / "outputs/perturbgen/test_local_adaptation"
    monkeypatch.setenv("PTM2CELLNET_PERTURBGEN_LOCAL_OUTPUT_ROOT", str(local_output))
    monkeypatch.setenv("PTM2CELLNET_PERTURBGEN_LOCAL_DATASET", "datlinger2021_test_adaptation")
    monkeypatch.setenv(
        "PTM2CELLNET_PERTURBGEN_LOCAL_TARGETS",
        str(PROJECT_ROOT / "configs/integration/perturbgen_targets/lck.csv"),
    )
    monkeypatch.setenv(
        "PTM2CELLNET_PERTURBGEN_LOCAL_INPUT_H5AD",
        str(PROJECT_ROOT / "ref/Perturbgen-src/data/perturbgen_m0_smoke/datlinger2021_m0smoke.h5ad"),
    )
    monkeypatch.setenv(
        "PTM2CELLNET_PERTURBGEN_LOCAL_ENCODER_CKPT",
        str(
            PROJECT_ROOT
            / "perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
        ),
    )
    monkeypatch.setenv(
        "PTM2CELLNET_PERTURBGEN_LOCAL_GENE_MEDIAN",
        str(PROJECT_ROOT / "ref/Perturbgen-src/perturbgen/pp/gene_median_dict_gftokens_gc95M.pkl"),
    )
    monkeypatch.setenv(
        "PTM2CELLNET_PERTURBGEN_LOCAL_TOKEN_DICT",
        str(PROJECT_ROOT / "ref/Perturbgen-src/perturbgen/pp/token_dict_gftokens_gc95M.pkl"),
    )
    monkeypatch.setenv(
        "PTM2CELLNET_PERTURBGEN_LOCAL_GENE_MAPPING",
        str(PROJECT_ROOT / "ref/Perturbgen-src/perturbgen/pp/ensembl_mapping_dict_gc95M.pkl"),
    )
    config = load_pipeline_config(LOCAL_ADAPTATION_TEMPLATE)
    plans = build_stage_plans(config, project_root=PROJECT_ROOT)

    train_mask_argv = plans[1].argv
    train_decoder_argv = plans[2].argv
    assert train_mask_argv[train_mask_argv.index("--ckpt_every_n_epochs") + 1] == "2"
    assert train_decoder_argv[train_decoder_argv.index("--ckpt_every_n_epochs") + 1] == "2"
    tokenise_argv = plans[0].argv
    assert tokenise_argv[tokenise_argv.index("--gene_filtering_mode") + 1] == "hvg"
    assert tokenise_argv[tokenise_argv.index("--hvg_mode") + 1] == "after_tokenisation"
    assert tokenise_argv[tokenise_argv.index("--genes_to_include_path") + 1].endswith("lck.csv")
    assert tokenise_argv[tokenise_argv.index("--exclude_non_GF_genes") + 1] == "True"
    assert "--max_len" not in train_mask_argv
    assert "--tgt_vocab_size" not in train_mask_argv
    assert "--max_len" not in train_decoder_argv
    assert "--tgt_vocab_size" not in train_decoder_argv
    perturb_payload = plans[3].generated_files[0].payload
    assert perturb_payload["trainer"]["tgt_vocab_size"] == "auto"
    assert perturb_payload["trainer"]["max_seq_length"] == "auto"
    assert perturb_payload["datamodule"]["max_len"] == "auto"


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
        "perturb_config": {
            "data": {},
            "trainer": {"tgt_vocab_size": 2002, "max_seq_length": 1024},
            "datamodule": {},
            "model": {"ckpt_masking_path": "decoder.ckpt"},
        },
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


def _assert_ckpt_masking_path_rejected(tmp_path, monkeypatch, mutation):
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    mutation(config["stages"]["perturb"]["perturb_config"]["model"])

    with pytest.raises(PerturbGenConfigError, match="silently skips inference"):
        build_stage_plans(config, project_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda model: model.pop("ckpt_masking_path"), id="absent"),
        pytest.param(lambda model: model.update(ckpt_masking_path=None), id="none"),
        pytest.param(lambda model: model.update(ckpt_masking_path="  "), id="blank"),
    ],
)
def test_perturb_contract_ckpt_masking_path_matrix(tmp_path, monkeypatch, mutation):
    _assert_ckpt_masking_path_rejected(
        tmp_path, monkeypatch, mutation
    )


@pytest.mark.parametrize("field", ["tgt_vocab_size", "max_seq_length"])
def test_perturb_contract_requires_both_explicit_dimensions(tmp_path, monkeypatch, field):
    # val.py:53 takes the explicit branch only when BOTH fields are present;
    # with either missing it derives dims via lexicographic max(input_id).
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    del config["stages"]["perturb"]["perturb_config"]["trainer"][field]

    with pytest.raises(PerturbGenConfigError, match="lexicographic max"):
        build_stage_plans(config, project_root=PROJECT_ROOT)


def test_perturb_contract_datamodule_max_len_must_equal_trainer_base(tmp_path, monkeypatch):
    # val.py:216-218 adds +100/+50 buffers itself and rewrites
    # datamodule.max_len to the unbuffered base; the config must stay
    # consistent in base units.
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    config["stages"]["perturb"]["perturb_config"]["datamodule"]["max_len"] = 1124

    with pytest.raises(PerturbGenConfigError, match="base value"):
        build_stage_plans(config, project_root=PROJECT_ROOT)


def test_stage_dimension_consistency_rejects_tgt_vocab_drift(tmp_path, monkeypatch):
    # A checkpoint restored under a different tgt_vocab_size than it was
    # trained with fails with a tensor size mismatch (2026-08-23 M0 smoke).
    _template_env(tmp_path, monkeypatch)
    config = load_pipeline_config(TEMPLATE)
    config["stages"]["train_mask"]["args"]["tgt_vocab_size"] = 2002
    config["stages"]["perturb"]["perturb_config"]["trainer"]["tgt_vocab_size"] = 2004
    config["stages"]["perturb"]["perturb_config"]["trainer"]["max_seq_length"] = 1024
    config["stages"]["perturb"]["perturb_config"]["datamodule"]["max_len"] = 1024

    with pytest.raises(PerturbGenConfigError, match="tgt_vocab_size drift"):
        build_stage_plans(config, project_root=PROJECT_ROOT)
