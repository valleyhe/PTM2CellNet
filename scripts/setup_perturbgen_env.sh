#!/usr/bin/env bash
# Build the isolated Python 3.11 environment for PerturbGen M0 work.
#
# Rationale (project_repair_report_20260822.md §5, 2026-08-23 window):
#   the shared SSH_unit env is Python 3.12.13 and mixes conflicting pins
#   (numpy>=2 vs project numpy<2, scvi-tools/scGPT conflicts). PerturbGen's
#   own Dockerfile targets python:3.11; its pyproject requires >=3.11.
#   This script must NOT touch the shared environment.
#
# Prereq: conda on PATH (user-writable envs dir). No sudo/docker needed.
# Usage:  bash scripts/setup_perturbgen_env.sh
set -euo pipefail

ENV_NAME=perturbgen
PY_VER=3.11
PERTURBGEN_SRC="$(pwd)/ref/Perturbgen-src"

[ -d "$PERTURBGEN_SRC" ] || { echo "ERROR: $PERTURBGEN_SRC not found"; exit 1; }

echo "== [1/6] conda env: $ENV_NAME (python $PY_VER) =="
if conda env list | rg -q "^${ENV_NAME} "; then
    echo "env already exists, reusing"
else
    conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi
PIP="/home/scu/anaconda3/envs/perturbgen/bin/pip"
PY="/home/scu/anaconda3/envs/perturbgen/bin/python"

echo "== [2/6] locked runtime deps (M0 closure, versions from pyproject.toml) =="
# M0 needs: torch+lightning (ckpt load), transformers/datasets/evaluate (HF
# offline init), scanpy/anndata/h5py/loompy (data), click (CLI), wandb
# (WandbLogger, offline mode), sympy (stray import in perturbgen sources).
# NOT installed: mpi4py (no direct import), jax/deepspeed/ray (training-only
# dependencies, not needed for inference/perturb smoke), T_perturb (it is a
# directory name, not an importable package — see configs/paths.py).
"$PIP" install --no-input --resume-retries 200 --timeout 90 \
    "torch==2.5.1" "pytorch-lightning==2.4.0" "lightning-utilities==0.11.9" "torchmetrics==1.4.1" \
    "transformers==4.47.0" "tokenizers==0.21.0" "datasets==3.0.0" "evaluate==0.4.3" \
    "huggingface-hub==0.26.5" "safetensors==0.4.5" \
    "anndata==0.11" "scanpy==1.11.2" "h5py==3.12.1" "loompy==3.0.7" \
    "numpy==2.2.0" "pandas==2.2.3" "scipy==1.14.1" "scikit-learn==1.5.2" \
    "click==8.1.7" "pyyaml==6.0.2" "tqdm==4.67.1" "wandb==0.17.9" "sympy==1.13.1" \
    "matplotlib==3.9.3" "seaborn==0.13.2"

echo "== [3/6] implicit hard deps (top-level upstream imports, no requirements entry) =="
# Discovered by the 2026-08-23 M0-5 smoke (project_repair_report_20260823.md
# §2.2): perturbgen/Modules/transformer.py imports einops; trainer.py imports
# scvi.distributions; src/metric.py imports jax; src/optimal_transport.py
# imports ot (POT); evaluate.load('rouge') needs rouge_score/nltk at runtime.
"$PIP" install --no-input --resume-retries 200 --timeout 90 \
    "einops==0.8.2" \
    "scvi-tools==1.4.2" \
    "jax==0.10.2" "jaxlib==0.10.2" \
    "POT==0.9.7.post1" \
    "rouge-score==0.1.2" "nltk==3.10.3"

# Version ceilings discovered by the 2026-08-23 M2 train-mask smoke:
#   wandb 0.17.9 pb2 breaks with protobuf>=5, and tensorboard>=2.19 pb2
#   requires protobuf>=5.26 — the only working combo is the one pinned here.
#   P40 (CUDA capability 6.1) additionally needs TORCHDYNAMO_DISABLE=1 for
#   any training entry point (triton requires >= 7.0).
"$PIP" install --no-input --resume-retries 200 --timeout 90 \
    "protobuf==4.25.8" "tensorboard==2.18.0"

echo "== [4/6] geneformer (scmaskgit/src/utils.py top-level import, --no-deps) =="
# --no-deps: its full requirement closure conflicts with the pins above; only
# the import chain is needed (tdigest/peft/optuna/tensorboard below).
"$PIP" install --no-input --no-deps --resume-retries 200 --timeout 90 \
    "geneformer @ git+https://huggingface.co/ctheodoris/Geneformer@04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
"$PIP" install --no-input --resume-retries 200 --timeout 90 \
    "tdigest==0.5.2.2" "peft==0.20.0" "optuna==4.9.0" "tensorboard==2.21.0"
# NOTE: evaluate.load('rouge') additionally requires a one-time ONLINE warmup
# to populate the HF modules cache; offline runs then work. See
# project_repair_report_20260823.md §2.2.

echo "== [5/6] editable install of perturbgen (no deps; pins above are authoritative) =="
(cd "$PERTURBGEN_SRC" && "$PIP" install --no-deps -e .)

echo "== [6/6] M0 evidence collection =="
"$PY" scripts/collect_perturbgen_env_evidence.py

echo "activate with: conda activate $ENV_NAME"
