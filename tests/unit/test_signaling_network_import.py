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


def test_root_signaling_network_is_shim_of_canonical_impl():
    """TD-M1 / CODE-03 acceptance: root signaling_network.py must be a pure
    re-export shim, not a duplicate implementation.

    The canonical implementation lives in ``src/models/signaling_network.py``.
    The root file is permitted to exist ONLY as a backwards-compatibility
    shim that re-exports symbols from the canonical module. This test pins
    that policy so static scans never mis-flag it as a duplicate.
    """
    import inspect

    sys.modules.pop("signaling_network", None)
    shim = importlib.import_module("signaling_network")
    canonical = importlib.import_module("src.models.signaling_network")

    # Every public name exported by the shim must be the very same object as
    # the one in the canonical module (i.e. real re-export, not a copy).
    for name in shim.__all__:
        assert hasattr(shim, name), f"shim __all__ lists {name!r} but it is missing"
        assert hasattr(canonical, name), (
            f"canonical src.models.signaling_network is missing {name!r}"
        )
        assert getattr(shim, name) is getattr(canonical, name), (
            f"{name!r} in root shim is not the same object as in canonical impl — "
            "root signaling_network.py must only re-export, never redefine."
        )

    # Defensive belt: the shim source itself must not define a class or def
    # (it should only contain imports and re-exports). This catches a future
    # edit that accidentally reintroduces a duplicate implementation.
    source = inspect.getsource(shim)
    assert "class SignalingNetworkMapper" not in source, (
        "root signaling_network.py must not define SignalingNetworkMapper — "
        "move any new logic into src/models/signaling_network.py."
    )
    assert "class PTMNetworkAnalyzer" not in source
