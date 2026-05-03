"""Conftest for pathway integration tests — provides sspa mock with cleanup."""
import sys
import pytest
from unittest.mock import MagicMock


@pytest.fixture(scope="session", autouse=True)
def _sspa_mock_session():
    """Inject sspa mock into sys.modules before test collection imports pathway_integration."""
    mock_sspa = MagicMock()
    sys.modules['sspa'] = mock_sspa

    # Also patch the module-level variables in pathway_integration
    # so that SSPA_AVAILABLE=True and sspa is the mock (not None).
    import src.analysis.pathway_integration as _pi
    _pi.SSPA_AVAILABLE = True
    _pi.sspa = mock_sspa

    yield mock_sspa

    # Restore originals
    _pi.SSPA_AVAILABLE = False
    _pi.sspa = None
    sys.modules.pop('sspa', None)
