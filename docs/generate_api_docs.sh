#!/usr/bin/env bash
# Generate Sphinx API documentation from the src package tree.
# Run from the project root.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure the output directory exists.
mkdir -p docs/api

# Generate .rst files for all modules under src.
sphinx-apidoc -o docs/api src
