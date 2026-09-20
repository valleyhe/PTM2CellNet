"""Deterministic PTM-activity smoke bundle (engineering contract only).

This is not a phosphoproteome, not KSTAR, and not a biology PASS. It writes
§4.1-shaped site quantification plus the stub tables needed to exercise
stages 1/3/4/5 before a real PTM cohort or a KSTAR environment exists.

The activity stub keeps ``method=KSTAR`` so it satisfies
``primary_activity_method``, but ``method_version`` is the smoke-stub marker
and must never be treated as a real kinase-activity inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.analysis.ptm_activity import PTM_INPUT_REQUIRED_COLUMNS

SMOKE_COHORT_ID = "PTM_SMOKE_20260918"
SMOKE_NETWORK_RELEASE = "smoke-2026-09-18"
SMOKE_METHOD_VERSION = "smoke-stub-20260918"
SMOKE_SOURCE_DATASET = "PTM2CellNet_smoke"
SMOKE_LINEAGE_BOUNDARY = "smoke_only"
KSTAR_METHOD_NAME = "KSTAR"
CONTRAST = "disease-minus-normal"

#: UniProt / symbol / Ensembl for mapped smoke proteins (canonical map, not guessed).
MAPPED_PROTEINS: tuple[tuple[str, str, str], ...] = (
    ("P49841", "GSK3B", "ENSG00000082701"),
    ("Q00535", "CDK5", "ENSG00000164885"),
    ("P31749", "AKT1", "ENSG00000142208"),
    ("P10636", "MAPT", "ENSG00000186868"),
    ("P05067", "APP", "ENSG00000142192"),
    ("P56817", "BACE1", "ENSG00000186318"),
    ("P28482", "MAPK1", "ENSG00000100030"),
    ("P02649", "APOE", "ENSG00000130203"),
)
NETWORK_ONLY: tuple[tuple[str, str, str], ...] = (
    ("Q9NQB0", "TCF7L2", "ENSG00000148737"),
    ("Q12778", "FOXO1", "ENSG00000150907"),
)
UNMAPPED_PROTEIN = "FAKEPROT1"

NORMAL_DONORS = ("N1", "N2", "N3")
DISEASE_DONORS = ("D1", "D2", "D3")

#: residue, gene, baseline linear intensity, disease fold vs normal.
SITES: tuple[tuple[str, str, float, float], ...] = (
    ("S9", "GSK3B", 1.0, 2.2),
    ("Y216", "GSK3B", 1.2, 0.55),
    ("Y15", "CDK5", 1.0, 0.5),
    ("S473", "AKT1", 1.0, 1.8),
    ("S396", "MAPT", 1.4, 0.6),
    ("T668", "APP", 1.0, 1.7),
    ("S498", "BACE1", 1.1, 0.7),
    ("T185", "MAPK1", 1.0, 1.5),
)

PROTEIN_BY_SYMBOL = {symbol: (protein_id, ensembl_id) for protein_id, symbol, ensembl_id in MAPPED_PROTEINS}


@dataclass(frozen=True)
class SmokeBundle:
    """Paths written by :func:`write_smoke_bundle`."""

    output_dir: Path
    paths: dict[str, Path]
    lineage: dict[str, Any]


def _protein_lookup() -> dict[str, tuple[str, str]]:
    return dict(PROTEIN_BY_SYMBOL)


def build_gene_map() -> pd.DataFrame:
    rows = [
        {"protein_id": protein_id, "gene_symbol": symbol, "ensembl_id": ensembl_id}
        for protein_id, symbol, ensembl_id in MAPPED_PROTEINS
    ]
    return pd.DataFrame(rows)


def build_site_quantification() -> pd.DataFrame:
    """§4.1 table: 3+3 donors, one replicate, one unmapped protein, one donor-less row."""

    lookup = _protein_lookup()
    rows: list[dict[str, Any]] = []
    for donor in (*NORMAL_DONORS, *DISEASE_DONORS):
        condition = "normal" if donor in NORMAL_DONORS else "disease"
        sample_id = f"{condition[0].upper()}_{donor}"
        for residue, symbol, baseline, fold in SITES:
            protein_id, _ = lookup[symbol]
            intensity = baseline if condition == "normal" else baseline * fold
            rows.append(
                {
                    "sample_id": sample_id,
                    "donor_id": donor,
                    "condition": condition,
                    "protein_id": protein_id,
                    "gene_symbol": symbol,
                    "residue": residue,
                    "ptm_type": "phosphorylation",
                    "ptm_value": round(intensity, 4),
                    "value_scale": "linear",
                    "ptm_qvalue": 0.02 if condition == "disease" else 0.2,
                    "total_protein_value": 4.0,
                    "species": "9606",
                    "source_dataset": SMOKE_SOURCE_DATASET,
                }
            )
    # Replicate of the first disease GSK3B S9 row — collapsed by replicate_policy=mean.
    first = next(
        row for row in rows if row["donor_id"] == "D1" and row["gene_symbol"] == "GSK3B" and row["residue"] == "S9"
    )
    replicate = dict(first)
    replicate["ptm_value"] = round(float(first["ptm_value"]) + 0.2, 4)
    rows.append(replicate)
    rows.append(
        {
            "sample_id": "U_UNMAPPED",
            "donor_id": "D1",
            "condition": "disease",
            "protein_id": UNMAPPED_PROTEIN,
            "gene_symbol": "UNK",
            "residue": "S1",
            "ptm_type": "phosphorylation",
            "ptm_value": 3.0,
            "value_scale": "linear",
            "ptm_qvalue": 0.01,
            "total_protein_value": 4.0,
            "species": "9606",
            "source_dataset": SMOKE_SOURCE_DATASET,
        }
    )
    rows.append(
        {
            "sample_id": "C_NODONOR",
            "donor_id": "",
            "condition": "disease",
            "protein_id": "P49841",
            "gene_symbol": "GSK3B",
            "residue": "S21",
            "ptm_type": "phosphorylation",
            "ptm_value": 1.5,
            "value_scale": "linear",
            "ptm_qvalue": 0.3,
            "total_protein_value": 4.0,
            "species": "9606",
            "source_dataset": SMOKE_SOURCE_DATASET,
        }
    )
    frame = pd.DataFrame(rows)
    missing = [column for column in PTM_INPUT_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"smoke PTM table missing columns: {missing}")
    return frame


def build_kstar_evidence(site_frame: pd.DataFrame) -> pd.DataFrame:
    """Donor-mean fold evidence table in the shape a KSTAR adapter would consume.

    This is a *handover*, not a KSTAR run: sites are labelled increased /
    decreased / unchanged from smoke quantification only.
    """

    mapped = site_frame[site_frame["protein_id"] != UNMAPPED_PROTEIN].copy()
    mapped = mapped[mapped["donor_id"].astype(str).str.len() > 0]
    mapped["ptm_value"] = pd.to_numeric(mapped["ptm_value"])
    grouped = (
        mapped.groupby(["protein_id", "gene_symbol", "residue", "condition"], sort=True)["ptm_value"]
        .mean()
        .unstack("condition")
    )
    rows: list[dict[str, Any]] = []
    for (protein_id, gene_symbol, residue), record in grouped.iterrows():
        normal = float(record["normal"])
        disease = float(record["disease"])
        fold = disease / normal if normal else float("nan")
        if fold >= 1.2:
            evidence_class = "increased"
        elif fold <= 1.0 / 1.2:
            evidence_class = "decreased"
        else:
            evidence_class = "unchanged"
        aa = residue[0]
        position = int(residue[1:])
        rows.append(
            {
                "site_id": f"{protein_id}_{residue}",
                "protein_id": protein_id,
                "gene_symbol": gene_symbol,
                "residue": residue,
                "aa": aa,
                "position": position,
                "mean_normal": round(normal, 6),
                "mean_disease": round(disease, 6),
                "disease_over_normal": round(fold, 6),
                "evidence_class": evidence_class,
                "kstar_increased": int(evidence_class == "increased"),
                "kstar_decreased": int(evidence_class == "decreased"),
                "contrast": CONTRAST,
                "lineage_boundary": SMOKE_LINEAGE_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def build_activity_stub() -> pd.DataFrame:
    """Placeholder kinase activities. ``method_version`` marks this as not KSTAR."""

    rows = [
        ("GSK3B", 2.0, "up", 12, 0.8),
        ("CDK5", -1.5, "down", 8, 0.6),
        ("AKT1", 1.2, "up", 10, 0.7),
    ]
    return pd.DataFrame(
        [
            {
                "activity_unit": "zscore",
                "regulator_id": regulator_id,
                "regulator_type": "kinase",
                "condition_or_contrast": CONTRAST,
                "activity_score": score,
                "activity_direction": direction,
                "activity_pvalue": 0.04,
                "activity_qvalue": 0.2,
                "n_substrates": n_substrates,
                "network_coverage": coverage,
                "method": KSTAR_METHOD_NAME,
                "method_version": SMOKE_METHOD_VERSION,
                "input_manifest": "ptm_input_manifest.json",
                "lineage_boundary": SMOKE_LINEAGE_BOUNDARY,
                "may_enter_lineage": False,
            }
            for regulator_id, score, direction, n_substrates, coverage in rows
        ]
    )


def build_signed_network() -> pd.DataFrame:
    """Tiny signed graph: kinase → TF (ppi) → gene (tf_regulation).

    This is not the OmniPath TF-only freeze. Kinase activity cannot be
    propagated on ``omnipath-2026-09-16`` until a kinase-substrate release exists.
    """

    edges = [
        ("GSK3B", "TCF7L2", "protein_protein", "+1", ""),
        ("CDK5", "TCF7L2", "protein_protein", "+1", ""),
        ("AKT1", "FOXO1", "protein_protein", "-1", ""),
        ("TCF7L2", "MAPT", "tf_regulation", "-1", "s1"),
        ("TCF7L2", "APOE", "tf_regulation", "+1", "s1"),
        ("TCF7L2", "APP", "tf_regulation", "+1", "s1"),
        ("FOXO1", "BACE1", "tf_regulation", "+1", "s1"),
    ]
    return pd.DataFrame(
        [
            {
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "effect_sign": sign,
                "site": site,
                "species": "9606",
                "evidence": "smoke_deterministic",
                "confidence": 1.0,
                "release": SMOKE_NETWORK_RELEASE,
            }
            for source_id, target_id, edge_type, sign, site in edges
        ]
    )


def build_network_id_map() -> pd.DataFrame:
    rows = [
        {"network_id": symbol, "gene_symbol": symbol, "ensembl_id": ensembl_id}
        for _, symbol, ensembl_id in (*MAPPED_PROTEINS, *NETWORK_ONLY)
    ]
    return pd.DataFrame(rows)


def build_deg_table() -> pd.DataFrame:
    """Smoke DEG aligned to the stub network, not the GSE174367 table."""

    # EX: source genes present so stage 5 can emit candidate rows under signed admission.
    ex_rows = [
        ("GSK3B", 0.4, 0.40, "up"),
        ("CDK5", -0.5, 0.41, "down"),
        ("AKT1", 0.3, 0.42, "up"),
        ("MAPT", -0.7, 0.43, "down"),
        ("APP", -0.4, 0.44, "down"),
        ("BACE1", -0.3, 0.45, "down"),
        ("APOE", 0.5, 0.46, "up"),
    ]
    inh_rows = [
        ("MAPT", 0.2, 0.60, "up"),
        ("APP", 0.3, 0.61, "up"),
    ]
    records: list[dict[str, Any]] = []
    lookup = _protein_lookup()
    for cell_type, rows in (("EX", ex_rows), ("INH", inh_rows)):
        for symbol, log2fc, fdr, direction in rows:
            _, ensembl_id = lookup[symbol]
            records.append(
                {
                    "cell_type": cell_type,
                    "ensembl_id": ensembl_id,
                    "gene_symbol": symbol,
                    "log2fc": log2fc,
                    "fdr": fdr,
                    "observed_direction": direction,
                    "n_normal_donors": 4,
                    "n_disease_donors": 8,
                }
            )
    return pd.DataFrame(records)


def build_source_proposals() -> pd.DataFrame:
    """Smoke proposals. Provenance is smoke, not literature."""

    rows = [
        ("P49841", 9, "S", "GSK3B", "up"),
        ("Q00535", 15, "Y", "CDK5", "down"),
        ("P31749", 473, "S", "AKT1", "up"),
    ]
    lookup = _protein_lookup()
    return pd.DataFrame(
        [
            {
                "protein_id": protein_id,
                "position": position,
                "ptm_type": "phosphorylation",
                "gene_symbol": symbol,
                "ensembl_id": lookup[symbol][1],
                "source_activity_id": symbol,
                "proposed_direction": direction,
                "site_probability": 0.9,
                "provenance": f"smoke:{SMOKE_COHORT_ID}",
                "aa": aa,
            }
            for protein_id, position, aa, symbol, direction in rows
        ]
    )


def build_embedding_vocab() -> dict[str, int]:
    return {
        "ENSG00000082701": 101,
        "ENSG00000164885": 102,
        "ENSG00000142208": 103,
    }


def write_research_config(path: Path, *, cohort_h5ad: Path) -> Path:
    text = "\n".join(
        [
            "# PTM smoke research config — engineering only, not the frozen AD yaml.",
            f"# ptm_cohort={SMOKE_COHORT_ID}; biology_pass must stay false.",
            "schema_version: ptm2cellnet.ptm-research-config/v1",
            "research_objective: association",
            "reference_axis: disease_minus_normal",
            f'contrast: "{CONTRAST}"',
            f"primary_activity_method: {KSTAR_METHOD_NAME}",
            "sensitivity_activity_method: PhosR",
            f'network_release: "{SMOKE_NETWORK_RELEASE}"',
            "cell_types: [EX, INH]",
            f"cohort_h5ad: {cohort_h5ad}",
            "cohort_pairing: between_donor",
            'species: "9606"',
            f"ptm_cohort: {SMOKE_COHORT_ID}",
            "deg_max_fdr: 0.05",
            "min_donors_per_state: 3",
            "replicate_policy: mean",
            "deg_donor_aggregation: pseudobulk_counts",
            "observed_admission_rule: signed_direction_without_fdr_cutoff",
            "kd_policy: merged_into_ko_out_of_scope",
            "public_perturbation_policy: out_of_scope",
            "propagation:",
            "  max_depth: 3",
            "  decay: 0.5",
            "  gene_edge_types: [tf_regulation]",
            "  max_paths_per_seed: null",
            "semantic_context:",
            '  context: "PTM smoke {cell_type} cells"',
            '  intervention: "KO"',
            '  comparison_baseline: "donor-level disease vs normal"',
            '  reference_axis: "disease_minus_normal"',
            "  research_objective: association",
            '  evidence_source: "PTM smoke stub (not KSTAR)"',
            f'  cohort: "{SMOKE_COHORT_ID}"',
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_cohort_h5ad(path: Path) -> Path:
    import anndata as ad
    import numpy as np

    obs = pd.DataFrame(
        {
            "cell_type": ["INH", "EX", "EX", "INH"],
            "donor": ["N1", "N1", "D1", "D1"],
            "state": ["normal", "normal", "disease", "disease"],
        },
        index=["cell0", "cell1", "cell2", "cell3"],
    )
    genes = [ensembl_id for _, _, ensembl_id in MAPPED_PROTEINS]
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=np.ones((4, len(genes)), dtype="float32"), obs=obs, var=var)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


def smoke_lineage() -> dict[str, Any]:
    return {
        "schema_version": "ptm2cellnet.ptm-smoke-lineage/v1",
        "ptm_cohort": SMOKE_COHORT_ID,
        "lineage_boundary": SMOKE_LINEAGE_BOUNDARY,
        "may_enter_lineage": False,
        "biology_pass": False,
        "kstar_ran": False,
        "activity_method_version": SMOKE_METHOD_VERSION,
        "network_release": SMOKE_NETWORK_RELEASE,
        "note": (
            "Deterministic PTM smoke for contract tests. Activity rows use method=KSTAR "
            "only to satisfy primary_activity_method; method_version marks a stub. "
            "Do not join this bundle into formal DAVF/PerturbGen lineage."
        ),
    }


def write_smoke_bundle(output_dir: str | Path) -> SmokeBundle:
    """Write the full smoke input bundle (does not run stage CLIs)."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    site = build_site_quantification()
    gene_map = build_gene_map()
    evidence = build_kstar_evidence(site)
    activity = build_activity_stub()
    network = build_signed_network()
    id_map = build_network_id_map()
    deg = build_deg_table()
    proposals = build_source_proposals()
    vocab = build_embedding_vocab()
    cohort = write_cohort_h5ad(inputs / "smoke_cohort.h5ad")
    config = write_research_config(inputs / "ptm_research_config.smoke.yaml", cohort_h5ad=cohort)

    paths = {
        "config": config,
        "ptm_site_quantification": inputs / "ptm_site_quantification.tsv",
        "gene_map": inputs / "gene_map.tsv",
        "kstar_evidence": inputs / "kstar_evidence_from_sites.tsv",
        "activity_stub": inputs / "ptm_activity.smoke_stub.tsv",
        "signed_network": inputs / "signed_network.smoke.tsv",
        "network_id_map": inputs / "network_id_map.tsv",
        "deg_table": inputs / "ad_deg.smoke.tsv",
        "source_proposals": inputs / "source_proposals.smoke.tsv",
        "cohort_h5ad": cohort,
        "embedding_vocab": inputs / "embedding_vocab.json",
        "lineage": directory / "smoke_lineage.json",
    }
    site.to_csv(paths["ptm_site_quantification"], sep="\t", index=False)
    gene_map.to_csv(paths["gene_map"], sep="\t", index=False)
    evidence.to_csv(paths["kstar_evidence"], sep="\t", index=False)
    activity.to_csv(paths["activity_stub"], sep="\t", index=False)
    network.to_csv(paths["signed_network"], sep="\t", index=False)
    id_map.to_csv(paths["network_id_map"], sep="\t", index=False)
    deg.to_csv(paths["deg_table"], sep="\t", index=False)
    proposals.to_csv(paths["source_proposals"], sep="\t", index=False)
    paths["embedding_vocab"].write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")
    lineage = smoke_lineage()
    paths["lineage"].write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return SmokeBundle(output_dir=directory, paths={key: path for key, path in paths.items()}, lineage=lineage)


def _run_stage(name: str, main: Any, argv: Sequence[str]) -> None:
    """Invoke a stage CLI without leaking its JSON stdout into the smoke payload."""

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        code = main(list(argv))
    if code != 0:
        detail = captured.getvalue().strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"smoke {name} failed{suffix}")


def run_smoke_pipeline(bundle: SmokeBundle) -> dict[str, Path]:
    """Run stages 1, 3, 4, 5 against the smoke bundle. Stage 2 (KSTAR) is skipped."""

    from scripts.build_celltype_candidate_specs import main as stage5_main
    from scripts.build_ptm_ad_intersections import main as stage4_main
    from scripts.build_ptm_global_gene_scores import main as stage3_main
    from scripts.run_ptm_activity import main as stage1_main

    outputs = bundle.output_dir / "pipeline"
    outputs.mkdir(parents=True, exist_ok=True)
    standardized = outputs / "standardized_ptm.tsv"
    input_manifest = outputs / "ptm_input_manifest.json"
    _run_stage(
        "stage 1 (run_ptm_activity)",
        stage1_main,
        [
            "--config",
            str(bundle.paths["config"]),
            "--input-tsv",
            str(bundle.paths["ptm_site_quantification"]),
            "--gene-map",
            str(bundle.paths["gene_map"]),
            "--output-tsv",
            str(standardized),
            "--manifest-output",
            str(input_manifest),
        ],
    )
    scores = outputs / "ptm_global_gene_scores.tsv"
    score_manifest = outputs / "gene_score_manifest.json"
    _run_stage(
        "stage 3 (build_ptm_global_gene_scores)",
        stage3_main,
        [
            "--config",
            str(bundle.paths["config"]),
            "--activity-tsv",
            str(bundle.paths["activity_stub"]),
            "--network-tsv",
            str(bundle.paths["signed_network"]),
            "--network-id-map",
            str(bundle.paths["network_id_map"]),
            "--output-tsv",
            str(scores),
            "--manifest-output",
            str(score_manifest),
        ],
    )
    intersections = outputs / "intersections"
    _run_stage(
        "stage 4 (build_ptm_ad_intersections)",
        stage4_main,
        [
            "--config",
            str(bundle.paths["config"]),
            "--gene-scores-tsv",
            str(scores),
            "--deg-table",
            str(bundle.paths["deg_table"]),
            "--output-dir",
            str(intersections),
        ],
    )
    specs = outputs / "specs"
    _run_stage(
        "stage 5 (build_celltype_candidate_specs)",
        stage5_main,
        [
            "--config",
            str(bundle.paths["config"]),
            "--source-proposals-tsv",
            str(bundle.paths["source_proposals"]),
            "--deg-table",
            str(bundle.paths["deg_table"]),
            "--target-set-manifest",
            str(intersections / "target_set_manifest.json"),
            "--context-h5ad",
            str(bundle.paths["cohort_h5ad"]),
            "--embedding-vocab",
            str(bundle.paths["embedding_vocab"]),
            "--output-dir",
            str(specs),
        ],
    )
    produced = {
        "standardized_ptm": standardized,
        "ptm_input_manifest": input_manifest,
        "gene_scores": scores,
        "gene_score_manifest": score_manifest,
        "intersections": intersections,
        "target_set_manifest": intersections / "target_set_manifest.json",
        "specs": specs,
        "build_summary": specs / "build_summary.json",
        "candidate_spec_EX": specs / "candidate_spec_EX.json",
    }
    report = {
        "ok": True,
        "biology_pass": False,
        "kstar_ran": False,
        "lineage_boundary": SMOKE_LINEAGE_BOUNDARY,
        "stages_run": ["1", "3", "4", "5"],
        "stage_2_skipped": "KSTAR not invoked; activity is smoke-stub",
        "outputs": {key: str(path) for key, path in produced.items()},
    }
    report_path = outputs / "smoke_pipeline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    produced["pipeline_report"] = report_path
    return produced


def write_and_run(output_dir: str | Path, *, run_pipeline: bool) -> dict[str, Any]:
    bundle = write_smoke_bundle(output_dir)
    payload: dict[str, Any] = {
        "ok": True,
        "biology_pass": False,
        "kstar_ran": False,
        "lineage_boundary": SMOKE_LINEAGE_BOUNDARY,
        "output_dir": str(bundle.output_dir),
        "inputs": {key: str(path) for key, path in bundle.paths.items()},
    }
    if run_pipeline:
        produced = run_smoke_pipeline(bundle)
        payload["pipeline"] = {key: str(path) for key, path in produced.items()}
    return payload


__all__ = [
    "KSTAR_METHOD_NAME",
    "SMOKE_COHORT_ID",
    "SMOKE_LINEAGE_BOUNDARY",
    "SMOKE_METHOD_VERSION",
    "SMOKE_NETWORK_RELEASE",
    "SmokeBundle",
    "build_activity_stub",
    "build_kstar_evidence",
    "build_site_quantification",
    "run_smoke_pipeline",
    "write_and_run",
    "write_smoke_bundle",
]
