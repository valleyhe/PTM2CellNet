import json
import sys
from pathlib import Path

import pytest

from src.integration.perturbgen.config_builder import build_stage_plans
from src.integration.perturbgen.contracts import (
    CandidateEvidence,
    DAVFDirectionEvidence,
)
from src.integration.perturbgen.env_guard import ExternalEnvironmentReport, validate_repo_roots
from src.integration.perturbgen.orchestrator import (
    PerturbGenInvocation,
    build_candidate_stage_plans,
)
from src.integration.perturbgen.runner import PerturbGenResumeError, PerturbGenRunner

pytestmark = pytest.mark.integration


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _mock_repo(root: Path) -> Path:
    _write(root / "pyproject.toml", "[tool.poetry]\nname='mock'\n")
    _write(root / "poetry.lock", "lock")
    _write(
        root / "perturbgen/__main__.py",
        (
            "from pathlib import Path\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "if '--sentinel_output' in args:\n"
            "    target = Path(args[args.index('--sentinel_output') + 1])\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            "    target.write_text('ok', encoding='utf-8')\n"
            "if '--ckpt_masking_path' in args:\n"
            "    assert Path(args[args.index('--ckpt_masking_path') + 1]).is_file()\n"
            "print('mock perturbgen cli')\n"
        ),
    )
    _write(root / "perturbgen/Perturb/val.py", "import sys, yaml\nfrom pathlib import Path\ncfg = yaml.safe_load(Path(sys.argv[2]).read_text(encoding='utf-8'))\nassert Path(cfg['model']['ckpt_masking_path']).is_file()\nout = Path(cfg['mock_output'])\nout.parent.mkdir(parents=True, exist_ok=True)\nout.write_text('ok', encoding='utf-8')\nprint('mock perturb')\n")
    return root


def _mock_project_script(root: Path) -> Path:
    return _write(
        root / "mock_export.py",
        (
            "from pathlib import Path\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "target = Path(args[args.index('--sentinel_output') + 1])\n"
            "target.parent.mkdir(parents=True, exist_ok=True)\n"
            "target.write_text('ok', encoding='utf-8')\n"
        ),
    )


def _config(tmp_path: Path, mock_repo: Path, export_script: Path) -> dict:
    output_root = tmp_path / "out"
    fingerprint_root = tmp_path / "fingerprints"
    token_input = _write(fingerprint_root / "input.h5ad")
    encoder_ckpt = _write(fingerprint_root / "encoder.ckpt")
    mapping = _write(fingerprint_root / "mapping.pkl")
    token_row = _write(fingerprint_root / "token_row.pkl")
    dependency = mock_repo / "pyproject.toml"

    def stage(name, *, driver, uses_gpu=False, extra=None):
        base = {
            "timeout_seconds": 30,
            "uses_gpu": uses_gpu,
            "expected_outputs": [{"path": f"{name}.ok", "kind": "file"}],
        }
        base.update(extra or {})
        base["driver"] = driver
        return base

    return {
        "schema_version": 1,
        "repo": {"perturbgen_repo": str(mock_repo)},
        "environment": {
            "python": sys.executable,
            "dependency_files": [str(dependency)],
            "asset_paths": [str(encoder_ckpt)],
        },
        "pipeline": {
            "perturbgen_commit": "aaaaaaa",
            "random_seed": 42,
            "output_root": str(output_root),
            "estimated_total_output_bytes": 1,
            "stage_versions": {
                "tokenise": 1,
                "train_mask": 1,
                "train_decoder": 1,
                "perturb": 1,
                "export_gene_embeddings": 1,
                "report": 1,
            },
            "dry_run_estimates": {},
        },
        "stages": {
            "tokenise": stage(
                "tokenise",
                driver="module",
                extra={
                    "module": "perturbgen",
                    "subcommand": "tokenise",
                    "args": {"sentinel_output": str(output_root / "tokenise" / "tokenise.ok")},
                    "fingerprint_paths": [str(token_input)],
                },
            ),
            "train_mask": stage(
                "train_mask",
                driver="module",
                uses_gpu=True,
                extra={
                    "module": "perturbgen",
                    "subcommand": "train-mask",
                    "args": {"sentinel_output": str(output_root / "train_mask" / "model" / "checkpoints" / "dynamic-mask.ckpt")},
                    "fingerprint_paths": [str(token_input)],
                    "expected_outputs": [{"name": "checkpoint", "path": "model/checkpoints", "discover_glob": "*.ckpt", "kind": "file"}],
                },
            ),
            "train_decoder": stage(
                "train_decoder",
                driver="module",
                uses_gpu=True,
                extra={
                    "module": "perturbgen",
                    "subcommand": "train-decoder",
                    "args": {
                        "ckpt_masking_path": "@artifact:train_mask:checkpoint",
                        "sentinel_output": str(output_root / "train_decoder" / "model" / "checkpoints" / "dynamic-decoder.ckpt"),
                    },
                    "fingerprint_paths": ["@artifact:train_mask:checkpoint"],
                    "expected_outputs": [{"name": "checkpoint", "path": "model/checkpoints", "discover_glob": "*.ckpt", "kind": "file"}],
                },
            ),
            "perturb": stage(
                "perturb",
                driver="perturb_script",
                uses_gpu=True,
                extra={
                    "generated_config_path": "generated/perturb.yaml",
                    "perturb_config": {
                        "mock_output": str(output_root / "perturb" / "results" / "dynamic-perturb.h5ad"),
                        "data": {},
                        "trainer": {
                            "mapping_dict_path": str(mapping),
                            "tokenid_to_rowid_path": str(token_row),
                            "tgt_vocab_size": 2002,
                            "max_seq_length": 1024,
                        },
                        "datamodule": {"max_len": 1024},
                        "model": {"ckpt_masking_path": "@artifact:train_decoder:checkpoint"},
                    },
                    "fingerprint_paths": [str(mapping), str(token_row), "@artifact:train_decoder:checkpoint"],
                    "expected_outputs": [{"name": "result_h5ad", "path": "results", "discover_glob": "*.h5ad", "kind": "file"}],
                },
            ),
            "export_gene_embeddings": stage(
                "export_gene_embeddings",
                driver="script",
                extra={
                    "script_root": "perturbgen",
                    "script_path": str(export_script),
                    "args": {"sentinel_output": str(output_root / "export_gene_embeddings" / "export_gene_embeddings.ok")},
                    "fingerprint_paths": [str(encoder_ckpt)],
                },
            ),
            "report": {
                "driver": "internal_report",
                "timeout_seconds": 30,
                "uses_gpu": False,
                "expected_outputs": [{"path": "pipeline_report.json", "kind": "json"}],
                "fingerprint_paths": [],
            },
        },
    }


def _patch_environment_probe(monkeypatch, mock_repo):
    report = ExternalEnvironmentReport(
        python_path=Path(sys.executable),
        python_version="3.11.0",
        roots=validate_repo_roots(perturbgen_repo_root=mock_repo),
        dependency_hashes={},
        asset_hashes={},
        perturbgen_commit="a" * 40,
    )
    monkeypatch.setattr(
        "src.integration.perturbgen.runner.probe_external_environment",
        lambda *args, **kwargs: report,
    )


def test_mocked_pipeline_runs_all_six_stages_and_supports_resume(tmp_path, monkeypatch):
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    _patch_environment_probe(monkeypatch, mock_repo)
    plans = build_stage_plans(_config(tmp_path, mock_repo, export_script))
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    first = runner.run_pipeline(plans, resume=False)
    assert [item.stage for item in first] == [
        "tokenise",
        "train_mask",
        "train_decoder",
        "perturb",
        "export_gene_embeddings",
        "report",
    ]
    report_path = tmp_path / "out" / "report" / "pipeline_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["stages"]

    second = runner.run_pipeline(plans, resume=True)
    assert all(item.reused for item in second)


def test_mocked_pipeline_rejects_resume_after_input_change(tmp_path, monkeypatch):
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    _patch_environment_probe(monkeypatch, mock_repo)
    config = _config(tmp_path, mock_repo, export_script)
    plans = build_stage_plans(config)
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    runner.run_pipeline(plans, resume=False)

    changed_input = Path(config["stages"]["train_mask"]["fingerprint_paths"][0])
    changed_input.write_text("changed", encoding="utf-8")

    with pytest.raises(PerturbGenResumeError, match="fingerprint changed"):
        runner.run_pipeline(plans, resume=True)


def test_mocked_pipeline_rejects_tampered_discovered_artifact(tmp_path, monkeypatch):
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    _patch_environment_probe(monkeypatch, mock_repo)
    plans = build_stage_plans(_config(tmp_path, mock_repo, export_script))
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")
    runner.run_pipeline(plans, resume=False)

    checkpoint = tmp_path / "out/train_mask/model/checkpoints/dynamic-mask.ckpt"
    checkpoint.write_text("tampered", encoding="utf-8")

    with pytest.raises(PerturbGenResumeError, match="upstream artifact train_mask:checkpoint changed"):
        runner.run_pipeline(plans, resume=True)


def test_candidate_stage_plans_isolate_both_paths_and_rewrite_target(tmp_path):
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    config = _config(tmp_path, mock_repo, export_script)
    config["stages"]["perturb"]["perturb_config"]["trainer"]["pert_tps"] = [1]
    config["stages"]["perturb"]["perturb_config"]["datamodule"]["pert_tps"] = [1]
    config["stages"]["train_mask"]["args"]["output_dir"] = str(
        Path(config["pipeline"]["output_root"]) / "train_mask" / "model"
    )
    candidate = CandidateEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=-1.0,
        observed_fdr=0.01,
        observed_direction="down",
        davf_action="oe",
        davf_score=None,
        davf_provenance="formal-checkpoint",
    )
    evidence = DAVFDirectionEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        predicted_direction="down",
        predicted_delta=-1.0,
        model_source="davf",
        checkpoint_provenance="formal-checkpoint",
        embedding_provenance="formal-embedding",
    )
    invocation = PerturbGenInvocation(
        intervention_type="KO",
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        target_token_id=17,
        perturbation_mode="overexpress",
        paths=("source_intervention", "within_state"),
        candidate=candidate,
        davf_evidence=evidence,
    )

    plans = build_candidate_stage_plans(
        config,
        invocation,
        output_root=tmp_path / "candidate-output",
        project_root=Path(__file__).resolve().parents[2],
    )

    assert [plan.name for plan in plans] == [
        "tokenise",
        "train_mask",
        "train_decoder",
        "source_intervention",
        "within_state",
        "export_gene_embeddings",
        "report",
    ]
    source_payload = plans[3].generated_files[0].payload
    within_payload = plans[4].generated_files[0].payload
    assert source_payload["trainer"]["genes_to_perturb"] == ["STAT3"]
    assert source_payload["trainer"]["perturbation_mode"] == "overexpress"
    assert source_payload["trainer"]["perturbation_sequence"] == ["src"]
    assert within_payload["trainer"]["perturbation_sequence"] == ["tgt"]
    assert str(tmp_path / "candidate-output") in plans[1].argv[-1]
    assert plans[3].output_dir.name == "source_intervention"
    assert plans[4].output_dir.name == "within_state"
