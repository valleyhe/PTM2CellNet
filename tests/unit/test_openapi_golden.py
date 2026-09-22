"""OpenAPI contract must match the versioned golden snapshot (TD-15).

The golden file (``tests/api/openapi_golden.json``) is the reviewed API
surface.  Any diff is a contract change: either fix the regression or, for an
intentional change, regenerate with ``python scripts/update_openapi_golden.py``
and review the diff in the same commit.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH = PROJECT_ROOT / "tests" / "api" / "openapi_golden.json"


def _sorted_paths(schema: dict) -> list[str]:
    return sorted(schema.get("paths", {}))


def test_openapi_matches_golden_snapshot() -> None:
    from src.api.app import create_app

    assert GOLDEN_PATH.is_file(), (
        f"missing OpenAPI golden snapshot: {GOLDEN_PATH}; run scripts/update_openapi_golden.py"
    )
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    current = create_app().openapi()

    golden_paths = _sorted_paths(golden)
    current_paths = _sorted_paths(current)
    assert current_paths == golden_paths, (
        "API path set changed vs golden:\n"
        f"  only in golden: {sorted(set(golden_paths) - set(current_paths))}\n"
        f"  only in current: {sorted(set(current_paths) - set(golden_paths))}\n"
        "If intentional, regenerate with scripts/update_openapi_golden.py."
    )
    assert current == golden, (
        "OpenAPI schema differs from the golden snapshot (schemas/params/responses). "
        "If intentional, regenerate with scripts/update_openapi_golden.py and review the diff."
    )


def test_golden_records_openapi_version_and_info() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["openapi"].startswith("3."), "golden must pin an OpenAPI 3.x document"
    assert set(("title", "version")) <= set(golden["info"]), "golden must record title and version"
