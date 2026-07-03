#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Legacy wrapper for PTM site prediction.

This compatibility entry point preserves existing ``python scripts/batch_predict.py``
usage, but the canonical script name is ``scripts/predict_ptm_sites.py`` because
the tool predicts PTM sites rather than cell states.

Scope boundary (avoid confusion with the API ``/batch_predict`` endpoint):
    - This CLI predicts **PTM sites** (which residues are modified) and is a
      thin re-export of ``scripts/predict_ptm_sites.py``.
    - The HTTP API ``/batch_predict`` endpoint (``src/api/routes/predictions.py``)
      performs **cell-state** batch prediction from sequences + PTM sites.
    They are distinct workflows sharing only a name; do not use this script
    expecting cell-state batch inference — use ``scripts/predict.py --input``
    or the API instead.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_ptm_sites import (  # noqa: F401
    BatchPTMPredictor,
    MultiTaskPTMPredictor,
    _build_parser,
    main,
    parse_args,
)


if __name__ == "__main__":
    main()
