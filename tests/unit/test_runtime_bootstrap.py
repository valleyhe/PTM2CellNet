import os
import sys
from pathlib import Path

from src.utils.runtime import bootstrap_runtime


def test_bootstrap_runtime_adds_local_pythonlibs_and_numba_cache(tmp_path, monkeypatch) -> None:
    project_root = tmp_path
    pythonlibs = project_root / ".pythonlibs"
    pythonlibs.mkdir()
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if str(pythonlibs) != entry])

    info = bootstrap_runtime(str(project_root))

    assert info["pythonlibs_added"] is True
    assert str(pythonlibs) in sys.path
    assert os.environ["NUMBA_CACHE_DIR"].startswith(str(project_root / "outputs" / "cache" / "numba"))
    assert Path(os.environ["NUMBA_CACHE_DIR"]).exists()
