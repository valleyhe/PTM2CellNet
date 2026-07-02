import pytest

from src.integration.contracts import GenePerturbationRequest
from src.integration.genki_adapter import GenKIAdapter

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_adapter_supports_hard_and_soft_modes() -> None:
    adapter = GenKIAdapter(ref_root="tests/fixtures/genki")

    hard = adapter.run(GenePerturbationRequest("TP53", "", "", -1, 1.0, "hard_ko"))
    soft = adapter.run(GenePerturbationRequest("TP53", "", "", -1, 0.6, "soft_ptm"))

    assert hard.mode == "hard_ko"
    assert soft.mode == "soft_ptm"
    assert hard.distance_score >= soft.distance_score


def test_soft_perturbation_cli_accepts_explicit_reference_files(tmp_path) -> None:
    gene_list = tmp_path / "genes.txt"
    network = tmp_path / "network.npy"
    counts = tmp_path / "counts.npy"
    gene_list.write_text("EGFR\nTP53\n", encoding="utf-8")
    np.save(network, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    np.save(counts, np.array([[2.0, 5.0], [1.0, 4.0]], dtype=float))

    config = {
        "integration": {
            "genki_ref_root": "unused",
            "gene_list_file": str(gene_list),
            "network_file": str(network),
            "counts_file": str(counts),
            "default_magnitude": 0.6,
        }
    }
    config_path = tmp_path / "soft.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_ptm_virtual_perturbation.py",
            "--config",
            str(config_path),
            "--gene",
            "TP53",
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert (output_dir / "hard_ko_results.json").exists()
    assert (output_dir / "soft_ptm_results.json").exists()
    assert (output_dir / "backend_info.json").exists()
    assert (output_dir / "hard_ko_gene_rank.csv").exists()
    assert (output_dir / "soft_ptm_gene_rank.csv").exists()
    assert (output_dir / "hard_ko_significance.json").exists()
    assert (output_dir / "soft_ptm_significance.json").exists()
    assert (output_dir / "hard_ko_generank.csv").exists()
    assert (output_dir / "soft_ptm_generank.csv").exists()
    assert (output_dir / "hard_ko_significant_genes.csv").exists()
    assert (output_dir / "soft_ptm_significant_genes.csv").exists()
    assert (output_dir / "hard_ko_gsea_rank.tsv").exists()
    assert (output_dir / "soft_ptm_gsea_rank.tsv").exists()
    assert (output_dir / "perturbation_summary.json").exists()
    assert (output_dir / "perturbation_summary.md").exists()


def test_soft_perturbation_cli_accepts_genki_source_backend(tmp_path) -> None:
    pytest.importorskip("torch_geometric")
    import anndata as ad
    import scipy.sparse as sp

    adata_path = tmp_path / "mini.h5ad"
    grn_dir = tmp_path / "GRNs"
    grn_dir.mkdir()
    adata = ad.AnnData(sp.csr_matrix(np.array([[-1.0, 0.5, 1.0], [0.2, -0.3, 0.1]], dtype=float)))
    adata.var_names = ["EGFR", "TP53", "BAX"]
    adata.layers["norm"] = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=float)
    adata.write_h5ad(adata_path)
    sp.save_npz(grn_dir / "pcNet.npz", sp.csr_matrix(np.array([[0.0, 0.9, 0.2], [0.9, 0.0, 0.8], [0.2, 0.8, 0.0]], dtype=float)))

    config = {
        "integration": {
            "genki_ref_root": "ref/GenKI-master-src/GenKI-master",
            "adata_file": str(adata_path),
            "grn_file_dir": str(grn_dir),
            "pcnet_name": "pcNet",
            "cutoff": 0,
            "default_magnitude": 0.6,
        }
    }
    config_path = tmp_path / "soft_genki_source.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output_dir = tmp_path / "outputs_source"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_ptm_virtual_perturbation.py",
            "--config",
            str(config_path),
            "--gene",
            "TP53",
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    backend_info = Path(output_dir / "backend_info.json").read_text(encoding="utf-8")
    assert '"backend": "genki_source"' in backend_info


def test_soft_perturbation_cli_accepts_latent_vgae_scoring(tmp_path) -> None:
    pytest.importorskip("torch_geometric")
    import anndata as ad
    import scipy.sparse as sp

    adata_path = tmp_path / "mini_latent.h5ad"
    grn_dir = tmp_path / "GRNs"
    grn_dir.mkdir()
    adata = ad.AnnData(
        sp.csr_matrix(
            np.array(
                [
                    [-1.0, 0.5, 1.0, -0.2],
                    [0.2, -0.3, 0.1, 0.4],
                    [0.1, 0.2, -0.2, 0.3],
                    [-0.4, 0.7, -0.1, 0.2],
                ],
                dtype=float,
            )
        )
    )
    adata.var_names = ["EGFR", "TP53", "BAX", "MDM2"]
    adata.layers["norm"] = np.array(
        [
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 3.0, 4.0, 1.5],
            [1.5, 2.5, 3.5, 2.0],
            [1.2, 2.8, 2.9, 1.8],
        ],
        dtype=float,
    )
    adata.write_h5ad(adata_path)
    sp.save_npz(
        grn_dir / "pcNet.npz",
        sp.csr_matrix(
            np.array(
                [
                    [0.0, 0.9, 0.0, 0.0],
                    [0.9, 0.0, 0.8, 0.0],
                    [0.0, 0.8, 0.0, 0.3],
                    [0.0, 0.0, 0.3, 0.0],
                ],
                dtype=float,
            )
        ),
    )

    config = {
        "integration": {
            "genki_ref_root": "ref/GenKI-master-src/GenKI-master",
            "adata_file": str(adata_path),
            "grn_file_dir": str(grn_dir),
            "pcnet_name": "pcNet",
            "cutoff": 0,
            "default_magnitude": 0.6,
            "scoring_method": "latent_vgae",
            "trainer_epochs": 1,
            "trainer_seed": 0,
        }
    }
    config_path = tmp_path / "soft_latent.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output_dir = tmp_path / "outputs_latent"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_ptm_virtual_perturbation.py",
            "--config",
            str(config_path),
            "--gene",
            "TP53",
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    soft_result = Path(output_dir / "soft_ptm_results.json").read_text(encoding="utf-8")
    assert '"scoring_method": "latent_vgae"' in soft_result
