import importlib
import subprocess
import sys


def test_root_signaling_network_imports_successfully():
    sys.modules.pop("signaling_network", None)

    module = importlib.import_module("signaling_network")

    assert module.SignalingNetworkMapper.SIGNALING_PATHWAYS


def test_src_signaling_network_import_does_not_emit_variant_workflow_warning():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.models.signaling_network import PTMNetworkAnalyzer; print(PTMNetworkAnalyzer.__name__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0
    assert "variant_workflow module not available" not in combined_output


def test_variant_workflow_import_does_not_emit_circular_import_warning():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.analysis.variant_workflow import VariantEffectWorkflow; print(VariantEffectWorkflow.__name__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0
    assert "variant_workflow module not available" not in combined_output
    assert "partially initialized module 'src.models.signaling_network'" not in combined_output
