#!/usr/bin/env python
"""M0-5 spike: real-data tokenisation + real-checkpoint perturb inference smoke.

Runs inside the dedicated Python 3.11 ``perturbgen`` conda environment:

    /home/scu/anaconda3/envs/perturbgen/bin/python scripts/run_perturbgen_m0_smoke.py

Stages (all against the locked upstream source at ref/Perturbgen-src, commit
a9a937526574b70aaa5fbf592bb56f35765ca83c; upstream code is never modified):

1. prep     - subset DatlingerBock2021.h5ad (raw counts + ENSG vars) to
              control (reference) + LAT_2 (CRISPR knockout target) cells.
2. tokenise - invoke the official ``perturbgen tokenise`` CLI on the subset.
3. perturb  - invoke the official ``perturbgen.Perturb.val`` entrypoint with a
              minimal eval YAML pointing at the real user-provided checkpoint.
4. profile  - per-batch forward timing (P50/P95) using the same upstream
              components, plus peak RSS / GPU memory sampling throughout.
5. evidence - write outputs/perturbgen/spike/<run_id>/evidence.json.

Let-it-crash: any failure aborts the run with a nonzero exit; no fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "ref" / "Perturbgen-src"
# perturbgen/configs/paths.py resolves ROOT to ref/ (parents[3] of the package
# dir), so upstream data/results live under ref/T_perturb, not inside the repo.
T_PROJECT_ROOT = REPO_ROOT.parent / "T_perturb"
CKPT_PATH = PROJECT_ROOT / (
    "perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_"
    "batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
)
SOURCE_H5AD = PROJECT_ROOT / "data/raw/scperturb/DatlingerBock2021.h5ad"
PP_DIR = REPO_ROOT / "perturbgen" / "pp"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d") + "_m0_smoke"
EVIDENCE_DIR = PROJECT_ROOT / "outputs" / "perturbgen" / "spike" / RUN_ID

# Smoke workload contract (M0 minimal dual-path workload):
# reference/control cells + one CRISPR knockout condition targeting gene LCK.
# Control cells are restricted to LCK-expressing ones: the mask-mode KO
# semantics knock the gene out of cells that actually express it.
REF_CONDITION = "control"
TGT_CONDITION = "LCK_2"
PERTURB_GENE = "LCK"
N_REF_CELLS = 400
N_TGT_CELLS = 350  # kept below N_REF_CELLS so 'control' is the largest pairing group
N_HVG = 2000
GENES_MIN_CELLS = 50
DATASET_NAME = "datlinger2021_m0smoke"

OFFLINE_ENV = {
    **os.environ,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


class ResourceSampler:
    """Samples peak RSS of a process tree and total GPU memory."""

    def __init__(self, poll_seconds: float = 0.2):
        self.poll_seconds = poll_seconds
        self.peak_rss_bytes = 0
        self.peak_gpu_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _tree_rss(self, pid: int) -> int:
        try:
            parent = psutil.Process(pid)
            procs = [parent, *parent.children(recursive=True)]
        except psutil.Error:
            return 0
        total = 0
        for p in procs:
            try:
                total += p.memory_info().rss
            except psutil.Error:
                continue
        return total

    def _gpu_mib(self) -> int:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return int(out.stdout.strip().splitlines()[0])
        except (subprocess.SubprocessError, ValueError, IndexError):
            return 0

    def _loop(self, pid: int) -> None:
        while not self._stop.is_set():
            rss = self._tree_rss(pid)
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            gpu = self._gpu_mib()
            if gpu > self.peak_gpu_mib:
                self.peak_gpu_mib = gpu
            self._stop.wait(self.poll_seconds)

    def start(self, pid: int) -> None:
        self._thread = threading.Thread(target=self._loop, args=(pid,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def stage_prep() -> dict:
    import pickle

    import numpy as np
    import scanpy as sc

    smoke_dir = REPO_ROOT / "data" / "perturbgen_m0_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    out_h5ad = smoke_dir / f"{DATASET_NAME}.h5ad"
    genes_csv = smoke_dir / "genes_to_include.csv"
    if out_h5ad.exists():
        out_h5ad.unlink()

    adata = sc.read_h5ad(SOURCE_H5AD)
    assert "ensembl_id" in adata.var.columns, "source h5ad must carry ensembl_id var"
    x_int = float((adata.X.data == np.round(adata.X.data)).mean())
    assert x_int == 1.0, f"source X is not raw counts (integer ratio {x_int})"

    with open(PP_DIR / "token_dict_gftokens_gc95M.pkl", "rb") as f:
        token_dict = pickle.load(f)
    with open(PP_DIR / "ensembl_mapping_dict_gc95M.pkl", "rb") as f:
        name2ensg = pickle.load(f)
    gene_ensg = name2ensg[PERTURB_GENE]
    assert gene_ensg in token_dict, f"{PERTURB_GENE} ({gene_ensg}) not in vocabulary"

    obs = adata.obs["perturbation"]
    gene_mask = (adata.var["ensembl_id"] == gene_ensg).to_numpy()
    assert gene_mask.sum() == 1, f"expected exactly one {PERTURB_GENE} column"
    gene_col = adata.X[:, gene_mask]
    gene_pos = np.asarray(gene_col.todense()).ravel() > 0

    ref_all = obs.index[(obs == REF_CONDITION) & gene_pos]
    assert len(ref_all) >= N_REF_CELLS, f"only {len(ref_all)} {REF_CONDITION} cells express {PERTURB_GENE}"
    ref_idx = ref_all[:N_REF_CELLS].tolist()
    tgt_idx = obs.index[obs == TGT_CONDITION][:N_TGT_CELLS].tolist()
    assert len(tgt_idx) == N_TGT_CELLS, f"only {len(tgt_idx)} {TGT_CONDITION} cells"
    subset = adata[ref_idx + tgt_idx].copy()

    lat_cells = N_REF_CELLS  # every selected reference cell expresses the gene
    assert lat_cells >= GENES_MIN_CELLS, f"{PERTURB_GENE} expressed in only {lat_cells} cells"

    subset.write_h5ad(out_h5ad)
    # gene list consumed by GF_tokenisation --genes_to_include_path; the
    # lower_bound_filter column marks rows eligible for forced inclusion.
    genes_csv.write_text(f"gene_name,lower_bound_filter\n{PERTURB_GENE},included\n")

    return {
        "source_h5ad": str(SOURCE_H5AD.relative_to(PROJECT_ROOT)),
        "source_shape": list(adata.shape),
        "subset_shape": list(subset.shape),
        "cells": {REF_CONDITION: N_REF_CELLS, TGT_CONDITION: N_TGT_CELLS},
        "x_integer_ratio": x_int,
        "perturb_gene": PERTURB_GENE,
        "perturb_gene_ensg": gene_ensg,
        "perturb_gene_expressed_cells": lat_cells,
        "subset_h5ad": file_record(out_h5ad),
        "genes_csv": file_record(genes_csv),
    }


def stage_tokenise() -> dict:
    tokenized_dir = T_PROJECT_ROOT / "tokenized_data" / DATASET_NAME
    if tokenized_dir.exists():
        shutil.rmtree(tokenized_dir)

    argv = [
        sys.executable,
        "-m",
        "perturbgen",
        "tokenise",
        "--h5ad_path",
        str(REPO_ROOT / "data" / "perturbgen_m0_smoke" / f"{DATASET_NAME}.h5ad"),
        "--dataset",
        DATASET_NAME,
        "--gene_filtering_mode",
        "hvg",
        "--hvg_mode",
        "after_tokenisation",
        "--var_list",
        "perturbation",
        "--pairing_mode",
        "random",
        "--time_obs",
        "perturbation",
        "--reference_time",
        REF_CONDITION,
        "--time_point_order",
        REF_CONDITION,
        TGT_CONDITION,
        "--nproc",
        "4",
        "--src_mode",
        "Geneformer",
        "--n_hvg",
        str(N_HVG),
        "--gene_median_path",
        str(PP_DIR / "gene_median_dict_gftokens_gc95M.pkl"),
        "--token_dict_path",
        str(PP_DIR / "token_dict_gftokens_gc95M.pkl"),
        "--gene_mapping_path",
        str(PP_DIR / "ensembl_mapping_dict_gc95M.pkl"),
        "--genes_to_include_path",
        str(REPO_ROOT / "data" / "perturbgen_m0_smoke" / "genes_to_include.csv"),
        "--genes_to_include_min_cells",
        str(GENES_MIN_CELLS),
    ]
    start = time.perf_counter()
    proc = subprocess.run(argv, cwd=REPO_ROOT, env=OFFLINE_ENV, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"tokenise failed with exit {proc.returncode}")

    suffix = f"{N_HVG}_hvg"
    artifacts = {}
    for rel in [
        f"tokenized_data/{DATASET_NAME}/token_id_to_genename_{suffix}.pkl",
        f"tokenized_data/{DATASET_NAME}/tokenid_to_rowid_{suffix}.pkl",
        f"tokenized_data/{DATASET_NAME}/dataset_{suffix}_src/{REF_CONDITION}.dataset",
        f"tokenized_data/{DATASET_NAME}/dataset_{suffix}_tgt/1_{TGT_CONDITION}.dataset",
    ]:
        path = T_PROJECT_ROOT / rel
        assert path.exists(), f"missing tokenise artifact: {rel}"
        if path.is_dir():
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            artifacts[rel] = {"path": rel, "size_bytes": size}
        else:
            artifacts[rel] = file_record(path)

    # The perturb target must be addressable in the tokenised gene space.
    import pickle

    with open(
        T_PROJECT_ROOT / f"tokenized_data/{DATASET_NAME}/token_id_to_genename_{suffix}.pkl",
        "rb",
    ) as f:
        rowid_to_gene = pickle.load(f)
    assert PERTURB_GENE in set(rowid_to_gene.values()), (
        f"{PERTURB_GENE} missing from tokenised gene space after HVG selection"
    )

    return {
        "argv": argv[1:],
        "elapsed_seconds": round(elapsed, 2),
        "artifacts": artifacts,
        "gene_space_size": len(rowid_to_gene),
    }


def build_eval_config() -> tuple[dict, Path]:
    suffix = f"{N_HVG}_hvg"
    troot = f"T_perturb/tokenized_data/{DATASET_NAME}"
    output_dir = T_PROJECT_ROOT / "res" / f"{DATASET_NAME}_perturb"
    config = {
        "data": {
            "src_dataset_file": f"{troot}/dataset_{suffix}_src/{REF_CONDITION}.dataset",
            "tgt_dataset_folder": f"{troot}/dataset_{suffix}_tgt",
            "tgt_adata_folder": f"{troot}/h5ad_pairing_{suffix}_tgt",
        },
        "trainer": {
            "d_model": 768,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 64,
            "dropout": 0,
            "mask_scheduler": "pow",
            "output_dir": str(output_dir),
            "mapping_dict_path": f"{troot}/token_id_to_genename_{suffix}.pkl",
            "tokenid_to_rowid_path": f"{troot}/tokenid_to_rowid_{suffix}.pkl",
            "encoder": "scmaskgit",
            "encoder_path": str(CKPT_PATH),
            "context_mode": False,
            "pos_encoding_mode": "time_pos_sin",
            "perturbation_mode": "mask",
            "genes_to_perturb": [PERTURB_GENE],
            "validation_mode": "inference",
            "perturbation_sequence": ["src"],
            "pert_tps": [1],
            "pred_tps": [1],
            "var_list": ["perturbation"],
            "use_count_decoder": False,
        },
        "datamodule": {
            "batch_size": 16,
            "num_workers": 4,
            "shuffle": False,
            "split": False,
            "pred_tps": [1],
            "var_list": ["perturbation"],
        },
        # Official val.py skips inference entirely when model.ckpt_masking_path
        # is None, and Lightning restores checkpoints with strict=True, so the
        # foundation checkpoint ('transformer.*' keys) must first be re-keyed
        # into a PerturberTrainer-shaped checkpoint (stage_convert).
        "model": {"precision": 16, "ckpt_masking_path": None},
    }
    return config, output_dir


def compute_seq_stats() -> dict:
    """Derive tgt_vocab_size / max_seq_length / max_len from tokenised data.

    Written explicitly into the eval config so official val.py takes its
    'if' branch (HSPC-sample style) instead of the 'else' branch whose
    max(input_id) over variable-length lists would be lexigraphic.
    """
    from datasets import load_from_disk

    troot = T_PROJECT_ROOT / "tokenized_data" / DATASET_NAME
    suffix = f"{N_HVG}_hvg"
    src = load_from_disk(str(troot / f"dataset_{suffix}_src" / f"{REF_CONDITION}.dataset"))
    tgt = load_from_disk(str(troot / f"dataset_{suffix}_tgt" / f"1_{TGT_CONDITION}.dataset"))
    max_id = 0
    max_len = 0
    for dataset in (src, tgt):
        ids = dataset["input_ids"]
        for row in ids:
            max_len = max(max_len, len(row))
    # Vocabulary size derives from tgt only (official val.py semantics).
    for row in tgt["input_ids"]:
        max_id = max(max_id, max(row) if row else 0)
    return {
        # Buffered values used when constructing modules directly; official
        # val.py adds the same +50/+100 buffers to the raw config values.
        "tgt_vocab_size": max_id + 50,
        "max_seq_length": max_len + 100,
        "base_tgt_vocab_size": max_id,
        "base_max_seq_length": max_len,
        "max_len": max_len,
        "src_rows": len(src),
        "tgt_rows": len(tgt),
    }


def _clear_shadowed_imports() -> None:
    """Stop scripts/evaluate.py from shadowing the evaluate HF package.

    The driver's script directory is sys.path[0]; perturbgen's Model.trainer
    does `import evaluate` expecting the site-packages package.
    """
    script_dir = str(Path(__file__).resolve().parent)
    while script_dir in sys.path:
        sys.path.remove(script_dir)
    for name in ("evaluate", "datasets"):
        mod = sys.modules.get(name)
        if mod is not None and not hasattr(mod, "__path__"):
            del sys.modules[name]


def stage_convert() -> dict:
    """Export a self-consistent PerturberTrainer checkpoint for val.py.

    The foundation checkpoint is a standalone scmoscf encoder (all 162 keys
    belong to the MaskGIT encoder: its own token_embedding/decoder_block/
    decoder_fc) and carries no PerturbGen decoder weights. Official val.py
    therefore loads it via trainer.encoder_path (strict, verified upstream in
    scmaskgitwrapper) and expects model.ckpt_masking_path to point at a
    downstream-trained checkpoint, which does not exist until M2 train-mask
    runs on Gate-0 donor data. This stage exports the freshly built module
    (real encoder weights + seed-42-initialised decoder) so Lightning's
    strict restore matches by construction. Outputs are engineering smoke
    evidence only - decoder logits are untrained noise - and must not be
    interpreted scientifically (that is Gate-E / M4 territory).
    """
    import torch

    os.chdir(REPO_ROOT.parent)
    _clear_shadowed_imports()
    config, _ = build_eval_config()
    stats = compute_seq_stats()
    trainer_cfg = dict(config["trainer"])
    trainer_cfg["tgt_vocab_size"] = stats["tgt_vocab_size"]
    trainer_cfg["max_seq_length"] = stats["max_seq_length"]
    # val.py overrides trainer.n_genes with the tgt h5ad width; mirror it so
    # the exported checkpoint restores strictly.
    import anndata as sc_ad

    trainer_cfg["n_genes"] = sc_ad.read_h5ad(
        T_PROJECT_ROOT / "tokenized_data" / DATASET_NAME / f"h5ad_pairing_{N_HVG}_hvg_tgt" / f"1_{TGT_CONDITION}.h5ad",
        backed="r",
    ).shape[1]

    from perturbgen.Perturb.trainer import PerturberTrainer

    module = PerturberTrainer(
        condition_dict=None,
        n_total_tps=1,
        conditions=None,
        conditions_combined=None,
        **trainer_cfg,
    )

    # Encoder fidelity: every foundation tensor must appear, unmodified,
    # under pretrained_model.encoder_layers.model.* (scmaskgitwrapper loads
    # the encoder strictly; this guards against silent structural drift).
    raw = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    foundation = raw["state_dict"] if "state_dict" in raw else raw
    own = module.state_dict()
    checked = 0
    for key, tensor in foundation.items():
        own_key = f"pretrained_model.encoder_layers.model.{key.removeprefix('transformer.')}"
        assert own_key in own, f"module misses encoder tensor {own_key}"
        assert torch.equal(own[own_key], tensor), f"encoder weight mismatch at {own_key}"
        checked += 1
    assert checked == len(foundation), "encoder fidelity covered all foundation keys"

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    converted = EVIDENCE_DIR / "perturber_encoder_real_decoder_seed42.ckpt"
    torch.save(
        {"state_dict": own, "pytorch-lightning_version": __import__("pytorch_lightning").__version__},
        converted,
    )
    return {
        "stats": stats,
        "encoder_keys_checked": checked,
        "decoder_weights": "random init (seed 42); no downstream ckpt exists before M2 train-mask",
        "semantic_boundary": (
            "smoke evidence for the official config path and resource "
            "baseline only; perturbation outputs are not scientifically "
            "interpretable until the decoder is trained (Gate-0 -> M2 -> M4)"
        ),
        "converted_ckpt": str(converted),
    }


def stage_perturb() -> dict:
    config, output_dir = build_eval_config()
    stats = compute_seq_stats()
    config["trainer"]["tgt_vocab_size"] = stats["base_tgt_vocab_size"]
    config["trainer"]["max_seq_length"] = stats["base_max_seq_length"]
    config["datamodule"]["max_len"] = stats["max_len"]
    config["model"]["ckpt_masking_path"] = str(EVIDENCE_DIR / "perturber_encoder_real_decoder_seed42.ckpt")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    config_path = EVIDENCE_DIR / "perturb_eval.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    argv = [
        sys.executable,
        "-m",
        "perturbgen.Perturb.val",
        "--config",
        str(config_path),
    ]
    start = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        env=OFFLINE_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sampler = ResourceSampler()
    sampler.start(proc.pid)
    stdout, _ = proc.communicate()
    sampler.stop()
    elapsed = time.perf_counter() - start
    log_path = EVIDENCE_DIR / "perturb_stdout.log"
    log_path.write_text(stdout or "")
    if proc.returncode != 0:
        print(stdout[-6000:], file=sys.stderr)
        raise RuntimeError(f"perturb inference failed with exit {proc.returncode}")

    outputs = sorted(output_dir.glob("*.h5ad"))
    assert outputs, f"no perturbation adata outputs under {output_dir}"

    import anndata as sc_ad

    output_records = []
    for path in outputs:
        a = sc_ad.read_h5ad(path)
        output_records.append(
            {
                **file_record(path),
                "shape": list(a.shape),
                "obs_columns": list(a.obs.columns)[:10],
            }
        )

    return {
        "argv": argv[1:],
        "config_file": file_record(config_path),
        "elapsed_seconds": round(elapsed, 2),
        "peak_rss_bytes": sampler.peak_rss_bytes,
        "peak_gpu_mib": sampler.peak_gpu_mib,
        "outputs": output_records,
        "stdout_log": file_record(log_path),
    }


def stage_profile() -> dict:
    """Per-batch forward timing on the official components (no upstream edits)."""
    import torch
    from datasets import load_from_disk
    from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
    from perturbgen.Perturb.trainer import PerturberTrainer
    from perturbgen.src.utils import read_dataset_files

    os.chdir(REPO_ROOT.parent)  # config paths are relative to upstream ROOT (ref/)
    _clear_shadowed_imports()
    config, _ = build_eval_config()
    stats = compute_seq_stats()
    suffix = f"{N_HVG}_hvg"
    troot = T_PROJECT_ROOT / "tokenized_data" / DATASET_NAME

    src_dataset = load_from_disk(troot / f"dataset_{suffix}_src" / f"{REF_CONDITION}.dataset")
    tgt_datasets = read_dataset_disk(troot / f"dataset_{suffix}_tgt")
    tgt_adatas = read_dataset_files(str(troot / f"h5ad_pairing_{suffix}_tgt"), "h5ad")
    tgt_counts = {k: v.X for k, v in tgt_adatas.items()}

    trainer_cfg = dict(config["trainer"])
    trainer_cfg["tgt_vocab_size"] = stats["tgt_vocab_size"]
    trainer_cfg["max_seq_length"] = stats["max_seq_length"]
    trainer_cfg["n_genes"] = tgt_adatas["tgt_h5ad_t1"].shape[1]
    module = PerturberTrainer(
        condition_dict=None,
        n_total_tps=len(tgt_adatas),
        conditions=None,
        conditions_combined=None,
        **trainer_cfg,
    ).cuda()
    module.eval()

    data_module = PerturbGenDataModule(
        n_total_tps=len(tgt_adatas),
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        condition_keys=None,
        condition_encodings=None,
        tgt_counts_dict=tgt_counts,
        conditions=None,
        conditions_combined=None,
        use_weighted_sampler=False,
        **{**config["datamodule"], "max_len": stats["max_len"]},
    )
    data_module.setup(stage="test")  # trainer.test would do this implicitly
    loader = data_module.test_dataloader()

    batch_times = []
    peak_gpu_alloc_mib = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items() if v is not None}
            t0 = time.perf_counter()
            _ = module.forward(batch, perturbation=True)
            torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - t0)
            peak_gpu_alloc_mib = torch.cuda.max_memory_allocated() / (1 << 20)

    assert batch_times, "profile produced zero batches"
    ordered = sorted(batch_times)

    def pct(p: float) -> float:
        k = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
        return ordered[k]

    return {
        "n_batches": len(batch_times),
        "batch_size": config["datamodule"]["batch_size"],
        "batch_seconds_mean": round(statistics.mean(batch_times), 4),
        "batch_seconds_p50": round(pct(0.50), 4),
        "batch_seconds_p95": round(pct(0.95), 4),
        "batch_seconds_min": round(ordered[0], 4),
        "batch_seconds_max": round(ordered[-1], 4),
        "torch_peak_allocated_mib": round(peak_gpu_alloc_mib, 1),
    }


def read_dataset_disk(folder: Path) -> dict:
    """Same keying contract as perturbgen.src.utils.read_dataset_files."""
    from datasets import load_from_disk

    out = {}
    for path in sorted(folder.glob("*.dataset")):
        out[f"tgt_dataset_t{path.name[0]}"] = load_from_disk(str(path))
    assert out, f"no .dataset files under {folder}"
    return out


def main() -> None:
    evidence = {
        "run_id": RUN_ID,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "M0 evidence item 5: real perturb smoke with the official config "
            "path plus resource baseline ([R2] 7.2 M0, [R3] 5 M0 window)"
        ),
        "python": sys.version,
        "perturbgen_repo": str(REPO_ROOT.relative_to(PROJECT_ROOT)),
        "checkpoint": file_record(CKPT_PATH),
    }

    for name, fn in [
        ("data", stage_prep),
        ("tokenise", stage_tokenise),
        ("convert", stage_convert),
        ("perturb", stage_perturb),
        ("profile", stage_profile),
    ]:
        print(f"=== stage {name} ===", flush=True)
        t0 = time.perf_counter()
        evidence[name] = fn()
        print(f"stage {name} done in {time.perf_counter() - t0:.1f}s", flush=True)

    evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / "evidence.json"
    with open(out, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"evidence written: {out}")


if __name__ == "__main__":
    main()
