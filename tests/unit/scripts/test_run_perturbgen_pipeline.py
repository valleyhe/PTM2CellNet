import pytest

from scripts.run_perturbgen_pipeline import _apply_path


def _config():
    return {
        "stages": {
            "perturb": {
                "perturb_config": {
                    "trainer": {"perturbation_sequence": ["src"], "pert_tps": [1]},
                    "datamodule": {"pert_tps": [1]},
                }
            }
        }
    }


def test_apply_source_path_removes_target_timepoints():
    config = _config()
    _apply_path(config, "source_intervention")
    perturb = config["stages"]["perturb"]["perturb_config"]
    assert perturb["trainer"]["perturbation_sequence"] == ["src"]
    assert "pert_tps" not in perturb["trainer"]
    assert "pert_tps" not in perturb["datamodule"]


def test_apply_within_state_uses_target_and_requires_pert_tps():
    config = _config()
    _apply_path(config, "within_state")
    assert config["stages"]["perturb"]["perturb_config"]["trainer"]["perturbation_sequence"] == ["tgt"]

    bad = _config()
    bad["stages"]["perturb"]["perturb_config"]["trainer"].pop("pert_tps")
    with pytest.raises(ValueError, match="requires.*pert_tps"):
        _apply_path(bad, "within_state")
