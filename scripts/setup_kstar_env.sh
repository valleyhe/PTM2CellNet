#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/environments/kstar/environment.yml"
CONDA_BIN="${CONDA_EXE:-conda}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
    echo "conda executable not found: ${CONDA_BIN}" >&2
    exit 1
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq kstar; then
    echo "kstar environment exists; running read-only verification"
else
    "${CONDA_BIN}" env create --file "${ENV_FILE}"
fi

KSTAR_PREFIX="$(${CONDA_BIN} run --name kstar bash -c 'printf "%s" "${CONDA_PREFIX:?}"')"
KSTAR_PYTHON="${KSTAR_PREFIX}/bin/python"
if [[ ! -x "${KSTAR_PYTHON}" ]]; then
    echo "kstar Python executable not found: ${KSTAR_PYTHON}" >&2
    exit 1
fi

EXPECTED_PYTHON="$(awk -F= '/^[[:space:]]*-[[:space:]]*python=/{print $2; exit}' "${ENV_FILE}")"
if [[ -z "${EXPECTED_PYTHON}" ]]; then
    echo "could not read pinned Python version from ${ENV_FILE}" >&2
    exit 1
fi

CONDA_PREFIX="${KSTAR_PREFIX}" CONDA_DEFAULT_ENV=kstar KSTAR_EXPECTED_PYTHON="${EXPECTED_PYTHON}" \
PTM2CELLNET_PROJECT_ROOT="${PROJECT_ROOT}" "${KSTAR_PYTHON}" - <<'PY'
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

import kstar
from kstar import config
from kstar import mapping

project_root = Path(os.environ["PTM2CELLNET_PROJECT_ROOT"])
sys.path.insert(0, str(project_root / "src" / "analysis"))
from kstar_adapter import build_kstar_input  # noqa: E402
from kstar_resources import (  # noqa: E402
    KSTAR_PINNED_PYTHON,
    KSTAR_PINNED_VERSION,
    KSTARResourceError,
    default_resource_hash_manifest,
    install_kstar_networks,
    verify_kstar_resource_files,
    verify_pinned_python_packages,
)


def _python_matches(actual: str, expected: str) -> bool:
    actual_parts = actual.split(".")
    expected_parts = expected.split(".")
    return actual_parts[: len(expected_parts)] == expected_parts


version = importlib.metadata.version("kstar")
if version != KSTAR_PINNED_VERSION:
    raise SystemExit(f"expected kstar {KSTAR_PINNED_VERSION}, found {version}")
expected_python = os.environ["KSTAR_EXPECTED_PYTHON"]
actual_python = ".".join(str(part) for part in sys.version_info[:3])
if not _python_matches(actual_python, expected_python):
    raise SystemExit(f"expected Python {expected_python}, found {actual_python}")
if actual_python != KSTAR_PINNED_PYTHON:
    raise SystemExit(f"expected frozen Python {KSTAR_PINNED_PYTHON}, found {actual_python}")
expected_prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
actual_prefix = Path(sys.prefix).resolve()
expected_executable = (actual_prefix / "bin" / "python").resolve()
actual_executable = Path(sys.executable).resolve()
if actual_prefix != expected_prefix:
    raise SystemExit(f"kstar prefix mismatch: sys.prefix={actual_prefix}, CONDA_PREFIX={expected_prefix}")
if actual_executable != expected_executable:
    raise SystemExit(
        f"kstar interpreter mismatch: sys.executable={actual_executable}, expected={expected_executable}"
    )
try:
    verify_pinned_python_packages(
        {
            "kstar": version,
            "pandas": importlib.metadata.version("pandas"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "requests": importlib.metadata.version("requests"),
            "tqdm": importlib.metadata.version("tqdm"),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "seaborn": importlib.metadata.version("seaborn"),
            "fpdf2": importlib.metadata.version("fpdf2"),
        }
    )
    verified = verify_kstar_resource_files(config.RESOURCE_DIR, manifest_path=default_resource_hash_manifest())
except KSTARResourceError as exc:
    raise SystemExit(str(exc)) from exc
print(f"python={sys.version.split()[0]}")
print(f"kstar={version}")
print(f"kstar_package={kstar.__file__}")
print(f"resource_dir={config.RESOURCE_DIR}")
print(f"resource_hashes_verified={sorted(verified)}")

# KSTAR ships NETWORKS.tar.gz and the extracted NETWORKS/ beside its package,
# not at the conda environment root. Reuse that location without overwriting it.
network_install_dir = Path(kstar.__file__).resolve().parent
try:
    network_root = install_kstar_networks(network_install_dir)
    config.update_configuration(network_dir=str(network_root), y_network_name="Default", st_network_name="Default")
except KSTARResourceError as exc:
    raise SystemExit(f"official KSTAR network installer failed: {exc}") from exc
print(f"network_asset_dir={network_install_dir}")
print(f"network_dir={config.NETWORK_DIR}")
print("configuration_check:")
config.check_configuration()
print("configuration_check=passed")

standardized_ptm = project_root / "outputs/ptm_activity/20260918_ptm_smoke/pipeline/standardized_ptm.tsv"
input_manifest = project_root / "outputs/ptm_activity/20260918_ptm_smoke/pipeline/ptm_input_manifest.json"
if not standardized_ptm.is_file() or not input_manifest.is_file():
    raise SystemExit(f"KSTAR mapping smoke input is missing: {standardized_ptm}, {input_manifest}")
input_frame, audit = build_kstar_input(
    standardized_ptm,
    input_manifest=input_manifest,
    case_condition="disease",
    reference_condition="normal",
    contrast="disease-minus-normal",
)
with tempfile.TemporaryDirectory(prefix="ptm2cellnet-kstar-smoke-") as output_dir:
    mapper = mapping.ExperimentMapper(
        input_frame,
        columns={"accession_id": "accession", "site": "site"},
        odir=output_dir,
        name="setup_smoke",
        data_columns=[
            "data:disease-minus-normal:increased",
            "data:disease-minus-normal:decreased",
        ],
        show_taskbar=False,
        auto_convert_ids=False,
        auto_format_peptides=False,
    )
    mapped = mapper.get_experiment()
if mapped.empty:
    raise SystemExit("official KSTAR ExperimentMapper returned no mapped rows")
print(f"mapping_rows={len(mapped)}")
print(f"mapping_columns={list(mapped.columns)}")
print(f"mapping_input_audit={audit.as_dict()}")
PY
