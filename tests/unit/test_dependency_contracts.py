"""Dependency-contract tests (N02/N15/N19, 2026-08-16).

Guards the single-source-of-truth rule established for dependency
declarations:

* ``setup.py`` ``install_requires`` is a mirror of
  ``requirements-core.txt`` (identical package set AND version pins);
* each extra (``pretrained``/``mamba``/``analysis``/``docs``/``dev``)
  matches the package set of its corresponding ``requirements-*.txt``
  file (minus what core already provides);
* heavy model stacks (transformers/lightning/mamba-ssm) never leak into
  the default install;
* security floors for requests/starlette/fastapi/uvicorn (N19) cannot
  regress.
"""

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP_PY = PROJECT_ROOT / "setup.py"
REQ_DIR = PROJECT_ROOT

_CORE_FILE = "requirements-core.txt"
_EXTRA_FILES = {
    "pretrained": "requirements-pretrained.txt",
    "mamba": "requirements-mamba.txt",
    "analysis": "requirements-analysis.txt",
    "docs": "requirements-docs.txt",
    "dev": "requirements-dev.txt",
}

_REQ_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*)$")


def _norm_name(name: str) -> str:
    return name.replace("_", "-").lower()


def _parse_req_line(line: str):
    """Return (norm_name, version_spec) or None for non-requirement lines."""
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        return None
    if stripped.startswith(("-r ", "-c ", "--", "git+", "http")):
        return None
    match = _REQ_LINE_RE.match(stripped)
    if not match:
        return None
    name, spec = match.group(1), match.group(2).strip()
    return _norm_name(name), spec


def parse_requirements_file(rel_path: str) -> dict:
    """Return {norm_name: version_spec} for a requirements file."""
    out: dict = {}
    for line in (REQ_DIR / rel_path).read_text(encoding="utf-8").splitlines():
        parsed = _parse_req_line(line)
        if parsed is not None:
            out[parsed[0]] = parsed[1]
    return out


def _extract_top_level_assignments(tree: ast.Module) -> dict:
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node.value
    return out


def _eval_ast_value(node, assignments: dict):
    """Evaluate the small AST subset used in setup.py (lists of strings / names)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_ast_value(elt, assignments) for elt in node.elts]
    if isinstance(node, ast.Name):
        return _eval_ast_value(assignments[node.id], assignments)
    if isinstance(node, ast.Dict):
        out = {}
        for key, value in zip(node.keys, node.values, strict=True):
            assert isinstance(key, ast.Constant) and isinstance(key.value, str)
            out[key.value] = _eval_ast_value(value, assignments)
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_ast_value(node.left, assignments) + _eval_ast_value(node.right, assignments)
    raise AssertionError(f"unsupported AST node in setup.py: {type(node).__name__}")


def _load_setup_declarations() -> tuple:
    """Return (install_requires, extras_require) as {norm_name: version_spec}."""
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    assignments = _extract_top_level_assignments(tree)
    setup_call = None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if getattr(node.value.func, "id", None) == "setup":
                setup_call = node.value
                break
    assert setup_call is not None, "setup() call not found"
    install: dict = {}
    extras: dict = {}
    for kw in setup_call.keywords:
        if kw.arg == "install_requires":
            for item in _eval_ast_value(kw.value, assignments):
                name, spec = _parse_req_line(item)
                assert name, f"unparseable install_requires entry: {item!r}"
                install[name] = spec
        elif kw.arg == "extras_require":
            raw = _eval_ast_value(kw.value, assignments)
            assert isinstance(raw, dict), "extras_require must be a dict literal"
            for group, deps in raw.items():
                specs = {}
                for item in deps:
                    name, spec = _parse_req_line(item)
                    assert name, f"unparseable extra entry: {item!r}"
                    specs[name] = spec
                extras[group] = specs
    return install, extras


@pytest.fixture(scope="module")
def declarations():
    return _load_setup_declarations()


def test_install_requires_is_exact_mirror_of_core(declarations) -> None:
    install, _ = declarations
    core = parse_requirements_file(_CORE_FILE)
    assert set(install) == set(core), (
        f"package-set drift: only-setup={sorted(set(install) - set(core))} "
        f"only-core={sorted(set(core) - set(install))}"
    )
    mismatched = {name for name in install if install[name] != core[name]}
    assert not mismatched, f"version-pin drift in install_requires vs core: {sorted(mismatched)}"


def test_heavy_stacks_not_in_default_install(declarations) -> None:
    install, _ = declarations
    for forbidden in ("transformers", "lightning", "mamba-ssm", "causal-conv1d", "scvi-tools", "anndata"):
        assert forbidden not in install, f"{forbidden} must stay out of install_requires (minimal core)"


def test_extras_match_requirement_files(declarations) -> None:
    _, extras = declarations
    for group, req_file in _EXTRA_FILES.items():
        req_specs = parse_requirements_file(req_file)
        core = parse_requirements_file(_CORE_FILE)
        # requirements-*.txt layer on top of core (-r core): subtract core packages.
        expected = {name: spec for name, spec in req_specs.items() if name not in core}
        actual = extras[group]
        assert set(actual) == set(expected), (
            f"extra[{group!r}] package drift: "
            f"only-extra={sorted(set(actual) - set(expected))} "
            f"only-file={sorted(set(expected) - set(actual))}"
        )


def test_api_extra_is_subset_of_install_requires(declarations) -> None:
    install, extras = declarations
    api = extras.get("api", {})
    assert set(api) <= set(install), f"api extra must stay within install_requires: {sorted(set(api) - set(install))}"


def test_all_extra_is_union_of_groups(declarations) -> None:
    _, extras = declarations
    groups = [g for g in extras if g != "all"]
    union = set()
    for group in groups:
        union |= set(extras[group])
    assert set(extras["all"]) == union, (
        f"all-extra drift: extra-only={sorted(set(extras['all']) - union)} "
        f"groups-only={sorted(union - set(extras['all']))}"
    )


def test_security_floor_versions_do_not_regress() -> None:
    """N19: CVE 修复下限。requests>=2.31 (CVE-2023-32681)、starlette>=0.40
    (CVE-2024-47874)、fastapi>=0.100、uvicorn>=0.20。"""
    core = parse_requirements_file(_CORE_FILE)
    assert core["requests"].startswith(">=2.31"), f"requests floor regressed: {core['requests']}"
    assert core["starlette"].startswith(">=0.40"), f"starlette floor regressed: {core['starlette']}"
    assert core["fastapi"].startswith(">=0.100"), f"fastapi floor regressed: {core['fastapi']}"
    assert core["uvicorn"].startswith(">=0.20"), f"uvicorn floor regressed: {core['uvicorn']}"


def test_python_version_targets_are_consistent() -> None:
    """N15: ruff target-version 与 mypy python_version、python_requires 一致（3.10）。"""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'target-version = "py310"' in pyproject, "ruff target-version must be py310"
    assert 'python_version = "3.10"' in pyproject, "mypy python_version must be 3.10"
    setup_text = SETUP_PY.read_text(encoding="utf-8")
    assert 'python_requires=">=3.10"' in setup_text, "python_requires must be >=3.10"
