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
    build_shared_prepare_plans,
)
from src.integration.perturbgen.runner import PerturbGenResumeError, PerturbGenRunner

pytestmark = pytest.mark.integration


def _semantic_context(route: str = "KO") -> dict[str, str]:
    return {
        "context": "disease",
        "intervention": route,
        "comparison_baseline": "normal",
        "reference_axis": "disease-minus-normal",
        "research_objective": "replication",
        "evidence_source": "donor_expression+davf_decode",
        "cohort": "formal",
    }


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
    _write(
        root / "perturbgen/Perturb/val.py",
        "import sys, yaml\nfrom pathlib import Path\ncfg = yaml.safe_load(Path(sys.argv[2]).read_text(encoding='utf-8'))\nassert Path(cfg['model']['ckpt_masking_path']).is_file()\nout = Path(cfg['mock_output']) if 'mock_output' in cfg else Path(cfg['trainer']['output_dir']) / 'dynamic-perturb.h5ad'\nout.parent.mkdir(parents=True, exist_ok=True)\nout.write_text('ok', encoding='utf-8')\nprint('mock perturb')\n",
    )
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
                    "args": {
                        "sentinel_output": str(
                            output_root / "train_mask" / "model" / "checkpoints" / "dynamic-mask.ckpt"
                        )
                    },
                    "fingerprint_paths": [str(token_input)],
                    "expected_outputs": [
                        {"name": "checkpoint", "path": "model/checkpoints", "discover_glob": "*.ckpt", "kind": "file"}
                    ],
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
                        "sentinel_output": str(
                            output_root / "train_decoder" / "model" / "checkpoints" / "dynamic-decoder.ckpt"
                        ),
                    },
                    "fingerprint_paths": ["@artifact:train_mask:checkpoint"],
                    "expected_outputs": [
                        {"name": "checkpoint", "path": "model/checkpoints", "discover_glob": "*.ckpt", "kind": "file"}
                    ],
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
                    "expected_outputs": [
                        {"name": "result_h5ad", "path": "results", "discover_glob": "*.h5ad", "kind": "file"}
                    ],
                },
            ),
            "export_gene_embeddings": stage(
                "export_gene_embeddings",
                driver="script",
                extra={
                    "script_root": "perturbgen",
                    "script_path": str(export_script),
                    "args": {
                        "sentinel_output": str(output_root / "export_gene_embeddings" / "export_gene_embeddings.ok")
                    },
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
        davf_score=0.5,
        davf_provenance="formal-checkpoint",
        proposed_direction="down",
        davf_predicted_direction="down",
        direction_gate_status="pass",
        semantic_context=_semantic_context(),
    )
    evidence = DAVFDirectionEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        predicted_direction="down",
        predicted_delta=-1.0,
        model_source="davf",
        checkpoint_provenance="formal-checkpoint",
        embedding_provenance="formal-embedding",
        confidence=0.5,
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
        semantic_context=_semantic_context(),
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


def _ko_mask_invocation():
    candidate = CandidateEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=1.0,
        observed_fdr=0.01,
        observed_direction="up",
        davf_action="ko",
        davf_score=0.5,
        davf_provenance="formal-checkpoint",
        proposed_direction="up",
        davf_predicted_direction="up",
        direction_gate_status="pass",
        semantic_context=_semantic_context(),
    )
    evidence = DAVFDirectionEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        predicted_direction="up",
        predicted_delta=1.0,
        model_source="davf",
        checkpoint_provenance="formal-checkpoint",
        embedding_provenance="formal-embedding",
        confidence=0.5,
    )
    return PerturbGenInvocation(
        intervention_type="KO",
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        target_token_id=17,
        perturbation_mode="mask",
        paths=("source_intervention", "within_state"),
        candidate=candidate,
        davf_evidence=evidence,
        semantic_context=_semantic_context(),
    )


def test_candidate_stage_plans_fan_out_seeds_and_sensitivity_modes(tmp_path):
    """§4.7 conditions 4/5: >=3 seeds and KO pad/delete sensitivity plans."""
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    config = _config(tmp_path, mock_repo, export_script)
    config["stages"]["perturb"]["perturb_config"]["trainer"]["pert_tps"] = [1]
    config["stages"]["perturb"]["perturb_config"]["datamodule"]["pert_tps"] = [1]
    config["stages"]["train_mask"]["args"]["output_dir"] = str(
        Path(config["pipeline"]["output_root"]) / "train_mask" / "model"
    )

    plans = build_candidate_stage_plans(
        config,
        _ko_mask_invocation(),
        output_root=tmp_path / "candidate-output",
        project_root=Path(__file__).resolve().parents[2],
        seeds=(0, 1, 2),
        sensitivity_modes=("pad", "delete"),
    )

    perturb_plans = [plan for plan in plans if plan.name in ("source_intervention", "within_state")]
    # 2 paths × (1 primary mask + 2 sensitivity modes) × 3 seeds = 18 plans
    assert len(perturb_plans) == 18
    combos = set()
    for plan in perturb_plans:
        payload = plan.generated_files[0].payload
        mode = payload["trainer"]["perturbation_mode"]
        assert plan.expected_outputs[0].discover_glob == f"*.h5ad" or plan.expected_outputs[0].discover_glob.endswith(
            f"_t{mode}.h5ad"
        )
        argv = list(plan.argv)
        seed = int(argv[argv.index("--seed") + 1])
        combos.add((plan.name, mode, seed))
    assert combos == {
        (path, mode, seed)
        for path in ("source_intervention", "within_state")
        for mode in ("mask", "pad", "delete")
        for seed in (0, 1, 2)
    }
    # Every combination writes into its own isolated directory.
    output_dirs = {str(plan.output_dir) for plan in perturb_plans}
    assert len(output_dirs) == 18


def test_candidate_stage_plans_reject_sensitivity_for_non_mask_primary(tmp_path):
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    config = _config(tmp_path, mock_repo, export_script)

    invocation = _ko_mask_invocation()
    object.__setattr__(invocation, "perturbation_mode", "overexpress")
    with pytest.raises(ValueError, match="sensitivity modes apply only to KO"):
        build_candidate_stage_plans(
            config,
            invocation,
            output_root=tmp_path / "candidate-output",
            sensitivity_modes=("pad",),
        )


def test_candidate_stage_plans_reject_duplicate_seeds(tmp_path):
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    config = _config(tmp_path, mock_repo, export_script)

    with pytest.raises(ValueError, match="unique non-negative"):
        build_candidate_stage_plans(
            config,
            _ko_mask_invocation(),
            output_root=tmp_path / "candidate-output",
            seeds=(0, 0),
        )


def test_sensitivity_invocation_contract_is_ko_only():
    invocation = _ko_mask_invocation()
    object.__setattr__(invocation, "perturbation_mode", "pad")
    object.__setattr__(invocation, "is_sensitivity", True)
    assert invocation.perturbation_mode == "pad"

    candidate = CandidateEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        cell_type="K562",
        ptm_context="STAT3:S12",
        observed_log2fc=1.0,
        observed_fdr=0.01,
        observed_direction="up",
        davf_action="ko",
        davf_score=0.5,
        davf_provenance="formal-checkpoint",
        proposed_direction="up",
        davf_predicted_direction="up",
        direction_gate_status="pass",
        semantic_context=_semantic_context("KD"),
    )
    evidence = DAVFDirectionEvidence(
        gene_symbol="STAT3",
        ensembl_id="ENSG00000168610",
        predicted_direction="up",
        predicted_delta=1.0,
        model_source="davf",
        checkpoint_provenance="formal-checkpoint",
        embedding_provenance="formal-embedding",
        confidence=0.5,
    )
    with pytest.raises(ValueError, match="KO-only"):
        PerturbGenInvocation(
            intervention_type="KD",
            gene_symbol="STAT3",
            ensembl_id="ENSG00000168610",
            target_token_id=17,
            perturbation_mode="pad",
            paths=("source_intervention", "within_state"),
            candidate=candidate,
            davf_evidence=evidence,
            semantic_context=_semantic_context("KD"),
            is_sensitivity=True,
        )


def test_shared_prepare_plans_run_once_and_candidates_reuse_artifacts(tmp_path, monkeypatch):
    """F-02: shared tokenise/train runs once; candidates execute perturb-only."""
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    _patch_environment_probe(monkeypatch, mock_repo)
    config = _config(tmp_path, mock_repo, export_script)
    config["stages"]["perturb"]["perturb_config"]["trainer"]["pert_tps"] = [1]
    config["stages"]["perturb"]["perturb_config"]["datamodule"]["pert_tps"] = [1]
    # The mock perturb script follows the rewritten trainer.output_dir like
    # the upstream val.py does.
    config["stages"]["perturb"]["perturb_config"].pop("mock_output", None)
    runner = PerturbGenRunner(gpu_lock_file=tmp_path / "gpu.lock")

    prepare_root = tmp_path / "shared" / "_prepare"
    prepare_plans = build_shared_prepare_plans(
        config,
        output_root=prepare_root,
    )
    assert [plan.name for plan in prepare_plans] == ["tokenise", "train_mask", "train_decoder"]
    assert all(plan.output_root == prepare_root.resolve() for plan in prepare_plans)

    prepare_results = runner.run_pipeline(prepare_plans, resume=False)
    registry = {(result.stage, name): path for result in prepare_results for name, path in result.artifacts.items()}
    assert ("train_mask", "checkpoint") in registry
    assert ("train_decoder", "checkpoint") in registry
    mask_checkpoint = Path(registry[("train_mask", "checkpoint")])
    decoder_checkpoint = Path(registry[("train_decoder", "checkpoint")])
    assert prepare_root.resolve() in mask_checkpoint.parents
    assert prepare_root.resolve() in decoder_checkpoint.parents

    plans = build_candidate_stage_plans(
        config,
        _ko_mask_invocation(),
        output_root=tmp_path / "shared" / "KO" / "ENSG00000168610",
        skip_prepare_stages=True,
        prepare_artifact_paths=registry,
    )
    assert [plan.name for plan in plans] == [
        "source_intervention",
        "within_state",
        "export_gene_embeddings",
        "report",
    ]
    for plan in plans[:2]:
        payload = plan.generated_files[0].payload
        assert payload["model"]["ckpt_masking_path"] == str(decoder_checkpoint.resolve())
    assert decoder_checkpoint.resolve() in plans[0].fingerprint_paths

    results = runner.run_pipeline(plans, resume=False)
    assert [result.stage for result in results] == [
        "source_intervention",
        "within_state",
        "export_gene_embeddings",
        "report",
    ]

    # Re-running the prepare root only reuses fingerprints; the candidate
    # plans never re-execute tokenise/training for the second invocation.
    second_prepare = runner.run_pipeline(prepare_plans, resume=True)
    assert all(result.reused for result in second_prepare)


def test_candidate_stage_plans_fail_when_shared_prepare_artifact_is_missing(tmp_path):
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    config = _config(tmp_path, mock_repo, export_script)
    config["stages"]["perturb"]["perturb_config"]["trainer"]["pert_tps"] = [1]
    config["stages"]["perturb"]["perturb_config"]["datamodule"]["pert_tps"] = [1]

    with pytest.raises(ValueError, match="shared prepare artifact was never produced"):
        build_candidate_stage_plans(
            config,
            _ko_mask_invocation(),
            output_root=tmp_path / "candidate-output",
            skip_prepare_stages=True,
            prepare_artifact_paths={},
        )


def test_candidate_stage_plans_keep_artifact_references_for_dry_run_preview(tmp_path):
    """Dry-run previews keep @artifact strings because nothing executes."""
    mock_repo = _mock_repo(tmp_path / "mock_repo")
    export_script = _mock_project_script(mock_repo)
    config = _config(tmp_path, mock_repo, export_script)
    config["stages"]["perturb"]["perturb_config"]["trainer"]["pert_tps"] = [1]
    config["stages"]["perturb"]["perturb_config"]["datamodule"]["pert_tps"] = [1]

    plans = build_candidate_stage_plans(
        config,
        _ko_mask_invocation(),
        output_root=tmp_path / "candidate-output",
        skip_prepare_stages=True,
    )
    assert [plan.name for plan in plans] == [
        "source_intervention",
        "within_state",
        "export_gene_embeddings",
        "report",
    ]
    payload = plans[0].generated_files[0].payload
    assert payload["model"]["ckpt_masking_path"] == "@artifact:train_decoder:checkpoint"
