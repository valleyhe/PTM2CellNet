"""Import hygiene guards for the canonical signaling_network module.

Historical note: the untracked root ``signaling_network.py`` re-export shim
was removed on 2026-08-16 (D3, commit-series of ``project_repair_report_20260816``
§4) after a repo-wide grep confirmed no references; the two tests that pinned
the shim's existence were removed with it. The canonical implementation is
``src/models/signaling_network.py``.
"""

import subprocess
import sys


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
