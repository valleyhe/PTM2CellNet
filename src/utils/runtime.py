"""Runtime bootstrap helpers for local vendored dependencies."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Dict, Optional


def bootstrap_runtime(project_root: Optional[str] = None) -> Dict[str, object]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    pythonlibs = root / ".pythonlibs"
    pythonlibs_added = False
    if pythonlibs.exists() and str(pythonlibs) not in sys.path:
        sys.path.insert(0, str(pythonlibs))
        pythonlibs_added = str(pythonlibs) in sys.path

    numba_cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", root / "outputs" / "cache" / "numba"))
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))

    return {
        "project_root": str(root),
        "pythonlibs_path": str(pythonlibs),
        "pythonlibs_added": pythonlibs_added,
        "numba_cache_dir": str(numba_cache_dir),
    }
