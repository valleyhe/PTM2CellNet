"""Metadata and processed-count utilities for the human IBD scRNA datasets.

The dataset scope and inclusion rules in this module intentionally follow
``IBD_dataset.md``.  In particular, the core atlas is limited to:

* GSE214695;
* GSE231993; and
* pretreatment, colonic samples from GSE282122.

GSE266616 is represented in the master metadata as an external validation
cohort, while ileal and post-treatment samples remain explicitly excluded
from the core atlas.

The module keeps metadata construction independent from matrix loading.  This
is important for auditing inclusion decisions before a large AnnData object is
created.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import pandas as pd


REQUIRED_METADATA_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "GSM",
    "library_id",
    "sample_id",
    "donor_id",
    "disease_group",
    "disease_subtype",
    "inflammation_status",
    "tissue",
    "intestinal_region",
    "treatment_status",
    "biologic_status",
    "timepoint",
    "response_status",
    "age",
    "sex",
    "chemistry",
    "cellranger_version",
    "analysis_role",
    "include_atlas",
    "include_primary_DE",
    "include_field_effect",
    "exclude_reason",
)


def _open_text(path: Path):
    """Open plain or gzip-compressed text files without guessing silently."""

    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _clean_value(value: Any) -> str:
    """Normalize a GEO value while preserving its biological wording."""

    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


def _key(value: str) -> str:
    """Return the stable key used for GEO characteristic fields."""

    return re.sub(r"\s+", " ", value.strip().lower())


def parse_geo_family_soft(path: Path) -> pd.DataFrame:
    """Parse sample-level records from a GEO family SOFT file.

    GEO stores one ``!Sample_title`` block per GSM.  Characteristics can use
    keys containing spaces and are therefore returned both as flattened
    columns and in ``characteristics_json``.  Repeated keys retain the first
    value in the flattened column and all values in the JSON payload.
    """

    with _open_text(path) as handle:
        text = handle.read()

    records: List[Dict[str, Any]] = []
    for raw_block in text.split("!Sample_title = ")[1:]:
        lines = raw_block.splitlines()
        if not lines:
            continue
        record: Dict[str, Any] = {"sample_title": _clean_value(lines[0])}
        characteristics: Dict[str, List[str]] = {}
        for line in lines[1:]:
            if line.startswith("!Sample_geo_accession = "):
                record["GSM"] = _clean_value(line.split(" = ", 1)[1])
            elif line.startswith("!Sample_characteristics_ch1 = "):
                value = _clean_value(line.split(" = ", 1)[1])
                if ":" in value:
                    raw_key, raw_value = value.split(":", 1)
                    characteristic_key = _key(raw_key)
                    characteristic_value = _clean_value(raw_value)
                else:
                    characteristic_key = "unstructured"
                    characteristic_value = value
                characteristics.setdefault(characteristic_key, []).append(characteristic_value)

        for characteristic_key, values in characteristics.items():
            record[characteristic_key] = values[0]
        record["characteristics_json"] = json.dumps(
            characteristics, ensure_ascii=False, sort_keys=True
        )
        if "GSM" not in record:
            raise ValueError(f"SOFT sample block has no GEO accession: {record['sample_title']}")
        records.append(record)

    if not records:
        raise ValueError(f"No sample records found in SOFT file: {path}")
    return pd.DataFrame(records)


def _base_record(
    *,
    dataset: str,
    gsm: str,
    donor_id: str,
    disease_group: str,
    disease_subtype: str,
    inflammation_status: str,
    tissue: str,
    intestinal_region: str,
    treatment_status: str,
    biologic_status: str,
    timepoint: str,
    response_status: str,
    age: str,
    sex: str,
    chemistry: str,
    analysis_role: str,
    include_atlas: bool,
    include_primary_de: bool,
    include_field_effect: bool = False,
    exclude_reason: str = "",
    sample_title: str = "",
    archive_member: str = "",
) -> Dict[str, Any]:
    """Build a record using the column contract from ``IBD_dataset.md``."""

    library_id = f"{dataset}_{gsm}"
    return {
        "dataset": dataset,
        "GSM": gsm,
        "library_id": library_id,
        "sample_id": library_id,
        "donor_id": donor_id,
        "disease_group": disease_group,
        "disease_subtype": disease_subtype,
        "inflammation_status": inflammation_status,
        "tissue": tissue,
        "intestinal_region": intestinal_region,
        "treatment_status": treatment_status,
        "biologic_status": biologic_status,
        "timepoint": timepoint,
        "response_status": response_status,
        "age": age,
        "sex": sex,
        "chemistry": chemistry,
        "cellranger_version": "not_reported",
        "analysis_role": analysis_role,
        "include_atlas": bool(include_atlas),
        "include_primary_DE": bool(include_primary_de),
        "include_field_effect": bool(include_field_effect),
        "exclude_reason": exclude_reason,
        "sample_title": sample_title,
        "archive_member": archive_member,
    }


def _build_gse214695_metadata(raw_dir: Path) -> List[Dict[str, Any]]:
    """Build metadata from the 18 annotated GSE214695 10x libraries."""

    pattern = re.compile(r"^(GSM\d+)_(HC|UC|CD)-(\d+)_matrix\.mtx\.gz$")
    records: List[Dict[str, Any]] = []
    for matrix_path in sorted(raw_dir.glob("*_matrix.mtx.gz")):
        match = pattern.match(matrix_path.name)
        if match is None:
            continue
        gsm, subtype, replicate = match.groups()
        sample_label = f"{subtype}{replicate}"
        disease_group = "HC" if subtype == "HC" else "IBD"
        records.append(
            _base_record(
                dataset="GSE214695",
                gsm=gsm,
                donor_id=f"GSE214695_{sample_label}",
                disease_group=disease_group,
                disease_subtype=subtype,
                inflammation_status="healthy" if subtype == "HC" else "not_reported",
                tissue="colonic_mucosa",
                intestinal_region="colon",
                treatment_status="not_reported",
                biologic_status="not_reported",
                timepoint="baseline",
                response_status="not_applicable",
                age="not_reported",
                sex="not_reported",
                chemistry="10x_3prime_not_reported",
                analysis_role="discovery",
                include_atlas=True,
                include_primary_de=True,
                sample_title=sample_label,
                archive_member=matrix_path.name,
            )
        )
    if len(records) != 18:
        raise ValueError(
            "GSE214695 requires 18 HC/UC/CD libraries; "
            f"found {len(records)} readable matrix filenames in {raw_dir}"
        )
    return records


def _build_gse231993_metadata(raw_dir: Path) -> List[Dict[str, Any]]:
    """Normalize the paired inflamed/noninflamed UC cohort."""

    soft_files = sorted(raw_dir.glob("*family.soft.gz"))
    if len(soft_files) != 1:
        raise ValueError(f"Expected one GSE231993 family SOFT file, found {soft_files}")
    frame = parse_geo_family_soft(soft_files[0])
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        sample_group = row.get("sample group", "")
        if sample_group == "HC":
            disease_group, subtype, inflammation = "HC", "HC", "healthy"
            role, primary = "discovery", True
        elif sample_group == "UC":
            disease_group, subtype, inflammation = "IBD", "UC", "inflamed"
            role, primary = "discovery", True
        elif sample_group == "UC-self control":
            disease_group, subtype, inflammation = "IBD", "UC", "noninflamed"
            role, primary = "paired_noninflamed", False
        else:
            raise ValueError(f"Unexpected GSE231993 sample group: {sample_group!r}")
        records.append(
            _base_record(
                dataset="GSE231993",
                gsm=str(row["GSM"]),
                donor_id=f"GSE231993_{row.get('individual', 'not_reported')}",
                disease_group=disease_group,
                disease_subtype=subtype,
                inflammation_status=inflammation,
                tissue=row.get("tissue", "colon"),
                intestinal_region="colon",
                treatment_status="not_reported",
                biologic_status="not_reported",
                timepoint="baseline",
                response_status="not_applicable",
                age="not_reported",
                sex="not_reported",
                chemistry="10x_3prime_not_reported",
                analysis_role=role,
                include_atlas=True,
                include_primary_de=primary,
                sample_title=str(row.get("sample_title", "")),
            )
        )
    if len(records) != 12:
        raise ValueError(f"GSE231993 requires 12 libraries; found {len(records)}")
    return records


def _build_gse282122_metadata(raw_dir: Path) -> List[Dict[str, Any]]:
    """Normalize the longitudinal, multi-site GSE282122 atlas."""

    soft_files = sorted(raw_dir.glob("*family.soft.gz"))
    if len(soft_files) != 1:
        raise ValueError(f"Expected one GSE282122 family SOFT file, found {soft_files}")
    frame = parse_geo_family_soft(soft_files[0])
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        disease = str(row.get("disease", "")).strip()
        if disease not in {"Healthy", "UC", "CD"}:
            raise ValueError(f"Unexpected GSE282122 disease: {disease!r}")
        subtype = "HC" if disease == "Healthy" else disease
        disease_group = "HC" if disease == "Healthy" else "IBD"
        site = str(row.get("site", "not_reported"))
        is_ileum = site.lower().replace("_", " ") == "terminal ileum"
        intestinal_region = "ileum" if is_ileum else "colon"
        inflammation_raw = str(row.get("inflammation", "not_reported"))
        inflammation = {
            "Healthy": "healthy",
            "Inflamed": "inflamed",
            "Non_Inflamed": "noninflamed",
        }.get(inflammation_raw, "not_reported")
        treatment_raw = str(row.get("treatment", "not_reported"))
        if treatment_raw == "Pre":
            treatment_status, timepoint = "pretreatment", "baseline"
        elif treatment_raw == "Post":
            treatment_status, timepoint = "post_treatment", "post_treatment"
        else:
            treatment_status, timepoint = "not_applicable", "baseline"

        core_colon = intestinal_region == "colon" and (
            disease == "Healthy" or treatment_raw == "Pre"
        )
        primary_contrast = core_colon and (
            disease == "Healthy" or inflammation == "inflamed"
        )
        if core_colon and primary_contrast:
            role = "discovery"
        elif core_colon:
            role = "paired_noninflamed"
        elif intestinal_region == "ileum":
            role = "ileal_excluded"
        else:
            role = "longitudinal_excluded"

        title = str(row.get("sample_title", ""))
        # The archive stores the same CID with a trailing "-reup" omitted.
        archive_member = title[:-5] if title.endswith("-reup") else title
        records.append(
            _base_record(
                dataset="GSE282122",
                gsm=str(row["GSM"]),
                donor_id=f"GSE282122_{row.get('patient', 'not_reported')}",
                disease_group=disease_group,
                disease_subtype=subtype,
                inflammation_status=inflammation,
                tissue="gut",
                intestinal_region=intestinal_region,
                treatment_status=treatment_status,
                biologic_status="not_reported",
                timepoint=timepoint,
                response_status="not_reported",
                age=str(row.get("age", "not_reported")),
                sex=str(row.get("sex", "not_reported")),
                chemistry=str(row.get("librarytype", "not_reported")),
                analysis_role=role,
                include_atlas=core_colon,
                include_primary_de=primary_contrast,
                sample_title=title,
                archive_member=archive_member,
            )
        )
    if len(records) != 216:
        raise ValueError(f"GSE282122 requires 216 libraries; found {len(records)}")
    return records


def _build_gse266616_metadata(raw_dir: Path) -> List[Dict[str, Any]]:
    """Normalize GSE266616 as ascending-colon external validation."""

    soft_files = sorted(raw_dir.glob("*family.soft.gz"))
    if len(soft_files) != 1:
        raise ValueError(f"Expected one GSE266616 family SOFT file, found {soft_files}")
    frame = parse_geo_family_soft(soft_files[0])
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        condition = str(row.get("condition", "")).strip()
        if condition not in {"control", "CD"}:
            raise ValueError(f"Unexpected GSE266616 condition: {condition!r}")
        subtype = "HC" if condition == "control" else "CD"
        disease_group = "HC" if condition == "control" else "IBD"
        tissue = str(row.get("tissue", "not_reported"))
        region = "colon" if tissue == "ascending_colon" else "ileum"
        sample_region = str(row.get("sample region", "not_reported"))
        inflammation = {
            "control": "healthy",
            "inf": "inflamed",
            "non": "noninflamed",
            "adjacent": "adjacent",
        }.get(sample_region, "not_reported")
        is_validation = region == "colon"
        records.append(
            _base_record(
                dataset="GSE266616",
                gsm=str(row["GSM"]),
                donor_id=f"GSE266616_{row.get('patient', 'not_reported')}",
                disease_group=disease_group,
                disease_subtype=subtype,
                inflammation_status=inflammation,
                tissue=tissue,
                intestinal_region=region,
                treatment_status=str(row.get("treatment (biological)", "not_reported")),
                biologic_status=str(row.get("treatment (biological)", "not_reported")),
                timepoint="baseline",
                response_status="not_applicable",
                age=str(row.get("age", "not_reported")),
                sex="not_reported",
                chemistry="10x_3prime_not_reported",
                analysis_role="validation" if is_validation else "ileal_excluded",
                include_atlas=False,
                include_primary_de=False,
                include_field_effect=False,
                exclude_reason="" if is_validation else "validation_scope_ascending_colon_only",
                sample_title=str(row.get("sample_title", "")),
                archive_member=str(row.get("sample_title", "")),
            )
        )
    if len(records) != 85:
        raise ValueError(f"GSE266616 GEO metadata contains 85 libraries; found {len(records)}")
    return records


def build_metadata_master(raw_root: Path | str = "data/raw") -> pd.DataFrame:
    """Build the complete, auditable metadata table for all four datasets."""

    root = Path(raw_root)
    records: List[Dict[str, Any]] = []
    records.extend(_build_gse214695_metadata(root / "gse214695"))
    records.extend(_build_gse231993_metadata(root / "gse231993"))
    records.extend(_build_gse282122_metadata(root / "gse282122"))
    records.extend(_build_gse266616_metadata(root / "gse266616"))
    frame = pd.DataFrame(records)
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Metadata contract missing columns: {missing}")
    frame = frame.sort_values(["dataset", "GSM"], kind="stable").reset_index(drop=True)
    if frame["GSM"].duplicated().any():
        duplicates = frame.loc[frame["GSM"].duplicated(keep=False), "GSM"].tolist()
        raise ValueError(f"GEO accessions must be unique across datasets: {duplicates[:10]}")
    return cast(pd.DataFrame, frame)


def write_metadata_outputs(
    metadata: pd.DataFrame,
    output_dir: Path | str = "data/processed/ibd",
) -> Tuple[Path, Path, Path]:
    """Write metadata, inclusion decisions, and a compact summary JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    master_path = output / "metadata_master.tsv"
    inclusion_path = output / "sample_inclusion.tsv"
    summary_path = output / "metadata_summary.json"

    ordered_columns = list(REQUIRED_METADATA_COLUMNS) + [
        column
        for column in ("sample_title", "archive_member", "characteristics_json")
        if column in metadata.columns
    ]
    metadata.loc[:, ordered_columns].to_csv(master_path, sep="\t", index=False)

    inclusion_columns = [
        "dataset",
        "GSM",
        "sample_id",
        "donor_id",
        "disease_group",
        "disease_subtype",
        "inflammation_status",
        "intestinal_region",
        "treatment_status",
        "timepoint",
        "analysis_role",
        "include_atlas",
        "include_primary_DE",
        "exclude_reason",
    ]
    metadata.loc[:, inclusion_columns].to_csv(inclusion_path, sep="\t", index=False)

    summary = {
        "total_libraries": int(len(metadata)),
        "by_dataset": {
            str(dataset): int(count)
            for dataset, count in metadata["dataset"].value_counts().sort_index().items()
        },
        "atlas_libraries": int(metadata["include_atlas"].sum()),
        "primary_de_libraries": int(metadata["include_primary_DE"].sum()),
        "by_role": {
            str(role): int(count)
            for role, count in metadata["analysis_role"].value_counts().sort_index().items()
        },
        "by_disease_group": {
            str(group): int(count)
            for group, count in metadata["disease_group"].value_counts().sort_index().items()
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return master_path, inclusion_path, summary_path


def load_metadata(path: Path | str) -> pd.DataFrame:
    """Load and validate a previously generated metadata table."""

    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {missing}")
    for column in ("include_atlas", "include_primary_DE", "include_field_effect"):
        frame[column] = frame[column].map({"True": True, "False": False, "1": True, "0": False})
        if frame[column].isna().any():
            raise ValueError(f"Metadata column {column} contains non-boolean values")
    return frame


__all__ = [
    "REQUIRED_METADATA_COLUMNS",
    "build_metadata_master",
    "load_metadata",
    "parse_geo_family_soft",
    "write_metadata_outputs",
]
