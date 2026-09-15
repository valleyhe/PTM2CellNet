"""PerturbGen 双路径报告。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def build_candidate_report_payload(
    decision: Any,
    *,
    manifest: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """稳定 JSON payload，保留 manifest 以便重放。"""

    payload = to_plain_object(decision)
    dual_payload = payload.get("dual_path") or payload
    path_decisions = dual_payload.get("path_decisions", [])
    candidate_payload = candidate or payload.get("candidate") or {}
    return {
        "candidate": dict(candidate_payload),
        "candidate_gene": dual_payload.get("candidate_gene") or candidate_payload.get("gene_symbol"),
        "intervention_type": dual_payload.get("intervention_type"),
        "verdict": payload.get("verdict"),
        "scientific_acceptance": dual_payload.get("scientific_acceptance", False),
        "evaluation_mode": dual_payload.get("evaluation_mode"),
        "evidence_class": dual_payload.get("evidence_class"),
        "q_value": dual_payload.get("q_value"),
        "reasons": list(payload.get("reasons", [])),
        "paths": list(path_decisions),
        "direction_gate": payload.get("direction_gate"),
        "manifest": dict(manifest),
    }


def build_candidate_summary_dataframe(payloads: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """候选层汇总表。"""

    rows: list[dict[str, Any]] = []
    for payload in payloads:
        path_map = {item.get("path"): item for item in payload.get("paths", [])}
        source = path_map.get("source_intervention", {})
        within = path_map.get("within_state", {})
        rows.append(
            {
                "candidate_gene": payload.get("candidate_gene"),
                "verdict": payload.get("verdict"),
                "scientific_acceptance": payload.get("scientific_acceptance", False),
                "evaluation_mode": payload.get("evaluation_mode"),
                "q_value": payload.get("q_value"),
                "source_verdict": source.get("verdict"),
                "within_state_verdict": within.get("verdict"),
                "source_median_rescue": source.get("median_rescue"),
                "within_state_median_rescue": within.get("median_rescue"),
                "reasons": ";".join(str(item) for item in payload.get("reasons", [])),
                "run_id": payload.get("manifest", {}).get("run_id"),
            }
        )
    return pd.DataFrame(rows)


def build_path_detail_dataframe(payload: Mapping[str, Any]) -> pd.DataFrame:
    """路径层明细表。"""

    rows: list[dict[str, Any]] = []
    for path_item in payload.get("paths", []):
        base_row = {
            "candidate_gene": payload.get("candidate_gene"),
            "path": path_item.get("path"),
            "verdict": path_item.get("verdict"),
            "primary_mode": path_item.get("primary_mode"),
            "median_rescue": path_item.get("median_rescue"),
            "worst_seed_rescue": path_item.get("worst_seed_rescue"),
            "seed_count": path_item.get("seed_count"),
            "evaluable_donors_min": path_item.get("evaluable_donors_min"),
            "donor_consistency_min": path_item.get("donor_consistency_min"),
            "reasons": ";".join(str(item) for item in path_item.get("reasons", [])),
        }
        seed_summaries = path_item.get("seed_summaries", [])
        if not seed_summaries:
            rows.append(base_row)
            continue
        for seed_item in seed_summaries:
            row = dict(base_row)
            row.update(
                {
                    "seed": seed_item.get("seed"),
                    "mode": seed_item.get("mode"),
                    "status": seed_item.get("status"),
                    "rescue_excl_target": seed_item.get("rescue_excl_target"),
                    "matched_null_count": seed_item.get("matched_null_count"),
                    "empirical_pvalue": seed_item.get("empirical_pvalue"),
                    "reason_code": seed_item.get("reason_code"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def render_candidate_markdown(payload: Mapping[str, Any]) -> str:
    """简短 markdown 摘要。"""

    lines = [
        "# PerturbGen Dual-Path Report",
        "",
        f"- Candidate: {payload.get('candidate_gene', 'unknown')}",
        f"- Verdict: {payload.get('verdict', 'unknown')}",
        f"- q-value: {payload.get('q_value')}",
        f"- Reasons: {', '.join(str(item) for item in payload.get('reasons', [])) or 'none'}",
        f"- Run ID: {payload.get('manifest', {}).get('run_id', 'unknown')}",
        "",
        "## Paths",
    ]
    for path_item in payload.get("paths", []):
        lines.extend(
            [
                f"- {path_item.get('path')}: {path_item.get('verdict')}",
                f"  - primary_mode: {path_item.get('primary_mode')}",
                f"  - median_rescue: {path_item.get('median_rescue')}",
                f"  - worst_seed_rescue: {path_item.get('worst_seed_rescue')}",
                f"  - reasons: {', '.join(str(item) for item in path_item.get('reasons', [])) or 'none'}",
            ]
        )
    return "\n".join(lines) + "\n"


def save_report_artifacts(
    payload: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """保存 json/csv/md 三类 artifact。"""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    summary_frame = build_candidate_summary_dataframe([payload])
    detail_frame = build_path_detail_dataframe(payload)
    markdown = render_candidate_markdown(payload)

    json_path = path / "candidate_report.json"
    csv_path = path / "candidate_summary.csv"
    detail_path = path / "path_details.csv"
    md_path = path / "candidate_report.md"

    json_path.write_text(
        json.dumps(to_plain_object(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_frame.to_csv(csv_path, index=False)
    detail_frame.to_csv(detail_path, index=False)
    md_path.write_text(markdown, encoding="utf-8")

    return {
        "json": str(json_path),
        "summary_csv": str(csv_path),
        "detail_csv": str(detail_path),
        "markdown": str(md_path),
    }


def to_plain_object(value: Any) -> Any:
    """Single JSON-ready serializer for report/lineage payloads.

    Handles Path (stringified), dataclasses (field-wise, so nested Paths and
    mapping keys are normalized too), mappings (keys stringified), and
    sequences.  Anything else is returned as-is: an exotic value then fails
    at ``json.dumps`` time instead of being silently stringified.
    """

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_plain_object(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_plain_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_object(item) for item in value]
    return value
