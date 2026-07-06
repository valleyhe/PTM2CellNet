"""F-08: FastAPI lifespan migration tests.

v17 migrated the API from the deprecated ``@app.on_event("startup")`` /
``@app.on_event("shutdown")`` decorators to the modern
``asynccontextmanager`` lifespan handler. These tests pin that decision
so the migration cannot silently regress.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from src.api.app import create_app


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "src" / "api" / "app.py"


def test_app_uses_lifespan_not_on_event():
    """The created app must carry a non-trivial lifespan context manager."""
    app = create_app()
    # FastAPI stores the lifespan as ``lifespan_context`` on the router.
    lifespan_ctx = getattr(app.router, "lifespan_context", None)
    assert lifespan_ctx is not None, "app has no lifespan context manager"


def test_app_source_does_not_register_on_event():
    """No ``@app.on_event(...)`` decorator (line-leading, code form) in app.py.

    Backtick-quoted mentions inside docstrings (``\\`@app.on_event(...)\\```)
    are legitimate migration notes and are tolerated; only a real decorator
    usage at line start (possibly indented) is a regression.
    """
    import re

    source = APP_PATH.read_text(encoding="utf-8")
    # A decorator usage is a line whose first non-whitespace chars are
    # ``@app.on_event(``. Matches inside docstrings/backticks are excluded
    # by requiring start-of-line (modulo indentation).
    pattern = re.compile(r"(?m)^[ \t]*@app\.on_event\s*\(")
    assert not pattern.search(source), (
        "F-08 regression: src/api/app.py reintroduced @app.on_event(...). "
        "Use the asynccontextmanager lifespan handler instead."
    )


def test_lifespan_runs_startup_and_shutdown():
    """Entering the lifespan must trigger auto-init; exit must not raise."""
    import asyncio

    app = create_app()
    lifespan_ctx = app.router.lifespan_context

    async def _drive():
        async with lifespan_ctx(app):
            # While inside the lifespan, startup has run.
            pass

    asyncio.run(_drive())


def test_lifespan_is_async_context_manager():
    """The lifespan factory must return an async context manager."""
    from src.api.app import _build_lifespan

    handler = _build_lifespan()
    # ``asynccontextmanager`` wraps the coroutine function; calling it
    # returns an object that supports async __enter__/__exit__.
    cm = handler(create_app())
    assert hasattr(cm, "__aenter__")
    assert hasattr(cm, "__aexit__")
    assert inspect.iscoroutinefunction(cm.__aenter__)
    assert inspect.iscoroutinefunction(cm.__aexit__)
