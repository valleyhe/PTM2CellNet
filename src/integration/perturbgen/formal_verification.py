"""Unified acceptance orchestration for formal Workflow A evidence.

Composes the existing Gate-4/5 verifiers into one reviewable verdict so a
formal run can be accepted or rejected from a single report instead of three
separate CLIs.  Checks are independent: each returns ``pass`` / ``fail`` /
``blocked`` (``blocked`` = required input not supplied yet), and the overall
verdict is ``fail`` if any check failed, ``blocked`` if any check is missing
its input, and ``pass`` only when every check passed.  This module never
re-implements a verifier that already exists; it only binds and records them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .frozen_cohort import load_frozen_manifest
from .null_generation import candidate_identity_from_e2e_report
from .null_selection import _validate_distribution_manifest  # noqa: SLF001

FORMAL_VERIFICATION_SCHEMA_VERSION = "ptm2cellnet.formal-workflow-a-verification/v1"
_E2E_SCHEMA_VERSION = "davf_perturbgen_e2e/v1"
_NULL_SCHEMA_VERSION = "perturbgen_null_distribution/v1"


class FormalVerificationError(ValueError):
    """Raised when verification inputs cannot be loaded or are malformed."""


def _load_json_mapping(value: Mapping[str, Any] | str | Path, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(str(value)).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalVerificationError(f"{name} is not readable JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise FormalVerificationError(f"{name} must contain a JSON object")
    return dict(payload)


def _check(status: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status}
    payload.update(extra)
    return payload


def _check_e2e_gate(e2e_report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    version = e2e_report.get("schema_version")
    if version != _E2E_SCHEMA_VERSION:
        return _check("fail", reasons=[f"unsupported schema_version {version!r}"])
    try:
        candidate = candidate_identity_from_e2e_report(e2e_report)
    except ValueError as exc:
        reasons.append(f"no_passing_invocation: {exc}")
        candidate = None
    runs = e2e_report.get("perturbgen_runs")
    if not isinstance(runs, list) or not runs:
        reasons.append("no_perturbgen_runs")
    evidence = e2e_report.get("statistical_evidence")
    if not isinstance(evidence, Mapping) or str(evidence.get("status", "")).strip() != "assembled":
        reasons.append("statistical_evidence_not_assembled")
    elif not bool(evidence.get("scientific_acceptance")):
        reasons.append("statistical_evidence_scientific_acceptance_false")
    payload = _check("fail" if reasons else "pass", reasons=reasons)
    if candidate is not None:
        payload["candidate"] = candidate
    return payload


def _check_frozen_cohort(frozen_manifest: str | Path | None) -> dict[str, Any]:
    if frozen_manifest is None:
        return _check("blocked", reasons=["frozen_manifest_not_supplied"])
    try:
        manifest = load_frozen_manifest(Path(str(frozen_manifest)))
    except (OSError, ValueError) as exc:
        return _check("fail", reasons=[f"frozen_manifest_invalid: {exc}"])
    try:
        manifest.audit_donor_leakage()
    except ValueError as exc:
        return _check("fail", reasons=[f"donor_leakage: {exc}"])
    return _check(
        "pass",
        cell_type=manifest.cell_type,
        n_train_donors=len(manifest.train_donors),
        n_held_out_donors=len(manifest.held_out_donors),
    )


def _check_matched_null(manifests: Sequence[Mapping[str, Any] | str | Path]) -> dict[str, Any]:
    if not manifests:
        return _check("blocked", reasons=["null_distribution_manifests_not_supplied"])
    entries: list[dict[str, Any]] = []
    reasons: list[str] = []
    for index, item in enumerate(manifests):
        label = f"null_distribution_manifests[{index}]"
        try:
            payload = _load_json_mapping(item, name=label)
            if payload.get("schema_version") != _NULL_SCHEMA_VERSION:
                raise FormalVerificationError(f"schema_version must be {_NULL_SCHEMA_VERSION!r}")
            validated = _validate_distribution_manifest(payload, required_count=99)
        except (FormalVerificationError, ValueError) as exc:
            reasons.append(f"{label}_invalid: {exc}")
            continue
        entries.append(
            {
                "candidate_ensembl_id": validated.get("candidate_ensembl_id"),
                "path": validated.get("path"),
                "mode": validated.get("mode"),
                "seed": validated.get("seed"),
                "value_count": len(validated["values"]),
                "required_count": validated.get("required_count"),
            }
        )
    payload = _check("fail" if reasons else "pass", reasons=reasons)
    payload["manifests"] = entries
    return payload


def _quality_from_eval_input(eval_input: Mapping[str, Any]) -> dict[str, Any]:
    candidates = eval_input.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FormalVerificationError("eval_input has no candidates")
    statuses: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise FormalVerificationError("eval_input candidates must be mappings")
        status = str(candidate.get("unperturbed_quality_status", "")).strip()
        source = str(candidate.get("unperturbed_quality_source", "")).strip()
        if not status:
            raise FormalVerificationError("eval_input candidate is missing unperturbed_quality_status")
        if source == "hand_filled":
            raise FormalVerificationError("formal eval_input must not carry hand-filled quality")
        statuses.append(status)
    if any(status != "pass" for status in statuses):
        return _check(
            "fail",
            reasons=[f"unperturbed_quality_status_{status}" for status in sorted(set(statuses)) if status != "pass"],
        )
    return _check("pass", candidates=len(statuses))


def _check_unperturbed_quality(
    quality_payload: Mapping[str, Any] | str | Path | None,
    eval_input: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if quality_payload is not None:
        try:
            payload = _load_json_mapping(quality_payload, name="quality_payload")
        except FormalVerificationError as exc:
            return _check("fail", reasons=[str(exc)])
        if payload.get("source") != "extract_unperturbed_quality_from_h5ad":
            return _check("fail", reasons=["quality_source_is_not_extract_unperturbed_quality_from_h5ad"])
        status = str(payload.get("status", "")).strip()
        if status != "pass":
            return _check("fail", reasons=[f"unperturbed_quality_status_{status or 'missing'}"])
        return _check("pass", candidates=1)
    if eval_input is not None:
        try:
            return _quality_from_eval_input(eval_input)
        except FormalVerificationError as exc:
            return _check("fail", reasons=[str(exc)])
    return _check("blocked", reasons=["quality_payload_and_eval_input_not_supplied"])


def _check_acceptance_replay(
    frozen_manifest_path: str | Path | None,
    eval_input: Mapping[str, Any] | None,
    report_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from .frozen_cohort import replay_verdicts, verify_eval_input_against_manifest

    if frozen_manifest_path is None or eval_input is None or report_manifest is None:
        missing = [
            name
            for name, value in (
                ("frozen_manifest", frozen_manifest_path),
                ("eval_input", eval_input),
                ("report_manifest", report_manifest),
            )
            if value is None
        ]
        return _check("blocked", reasons=[f"{name}_not_supplied" for name in missing])
    try:
        manifest = load_frozen_manifest(Path(str(frozen_manifest_path)))
        verification = verify_eval_input_against_manifest(manifest, dict(eval_input), require_formal=True)
        replay = replay_verdicts(report_manifest, manifest=manifest, eval_input=dict(eval_input), require_formal=True)
    except (OSError, ValueError) as exc:
        return _check("fail", reasons=[f"acceptance_replay_invalid: {exc}"])
    reasons: list[str] = []
    if not verification.get("covered"):
        reasons.append("eval_input_not_covered_by_frozen_manifest")
    if not verification.get("formal_plan_complete"):
        reasons.append("formal_plan_incomplete")
    if not replay.get("reproduced"):
        reasons.append("dual_path_replay_not_reproduced")
    if not replay.get("independent_h5ad_recomputed"):
        reasons.append("independent_h5ad_not_recomputed")
    payload = _check("fail" if reasons else "pass", reasons=reasons)
    payload["verification"] = verification
    payload["replay"] = replay
    return payload


def verify_formal_workflow_a(
    *,
    e2e_report: Mapping[str, Any] | str | Path,
    frozen_manifest: str | Path | None = None,
    null_distribution_manifests: Sequence[Mapping[str, Any] | str | Path] = (),
    quality_payload: Mapping[str, Any] | str | Path | None = None,
    eval_input: Mapping[str, Any] | str | Path | None = None,
    report_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Verify one formal Workflow A evidence bundle (Gate-4/5 acceptance)."""

    e2e_payload = _load_json_mapping(e2e_report, name="e2e_report")
    eval_payload = None if eval_input is None else _load_json_mapping(eval_input, name="eval_input")
    report_payload = None if report_manifest is None else _load_json_mapping(report_manifest, name="report_manifest")

    checks: dict[str, dict[str, Any]] = {
        "e2e_gate": _check_e2e_gate(e2e_payload),
        "frozen_cohort": _check_frozen_cohort(frozen_manifest),
        "matched_null": _check_matched_null(list(null_distribution_manifests)),
        "unperturbed_quality": _check_unperturbed_quality(quality_payload, eval_payload),
        "acceptance_replay": _check_acceptance_replay(frozen_manifest, eval_payload, report_payload),
    }

    statuses = [check["status"] for check in checks.values()]
    if "fail" in statuses:
        verdict = "fail"
    elif "blocked" in statuses:
        verdict = "blocked"
    else:
        verdict = "pass"
    return {
        "schema_version": FORMAL_VERIFICATION_SCHEMA_VERSION,
        "verdict": verdict,
        "checks": checks,
        "missing_inputs": [name for name, check in checks.items() if check["status"] == "blocked"],
    }
