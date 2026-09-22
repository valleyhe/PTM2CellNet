#!/usr/bin/env python3
"""Regenerate the versioned OpenAPI golden snapshot (TD-15).

Writes ``tests/api/openapi_golden.json`` from ``src.api.app.create_app()``.
The golden file is compared verbatim by ``tests/unit/test_openapi_golden.py``;
run this script only when an intentional API change lands, and review the diff.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH = PROJECT_ROOT / "tests" / "api" / "openapi_golden.json"


def main() -> int:
    from src.api.app import create_app

    schema = create_app().openapi()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"openapi golden written: {GOLDEN_PATH} (openapi {schema.get('openapi')}, info {schema.get('info')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
