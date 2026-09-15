"""Tests for deterministic matched-null selection and distributions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.integration.perturbgen.null_selection import (
    load_null_distribution_manifest,
    select_matched_nulls,
    summarize_null_distribution,
    write_null_selection_manifest,
)


TARGET = "ENSG00000000001"
CANDIDATES = ["ENSG00000000002", "ENSG00000000003"]
PATH = "source_intervention"


def _synthetic_cohort(null_count: int = 99) -> tuple[SimpleNamespace, list[str]]:
    null_ids = [f"ENSG{i:011d}" for i in range(4, 4 + null_count)]
    ids = [f"{TARGET}.7", f"{CANDIDATES[0]}.1", f"{CANDIDATES[1]}.2", *[f"{gene}.1" for gene in null_ids]]
    matrix = np.zeros((2, len(ids)), dtype=float)
    for index in range(len(ids)):
        value = float(index + 1)
        matrix[:, index] = (value, 0.0 if index % 2 else value)
    matrix[:, 0] = (2.0, 0.0)
    return SimpleNamespace(X=matrix, var=pd.DataFrame({"ensembl_id": ids}), var_names=ids), null_ids


def _records(count: int = 99) -> list[dict[str, object]]:
    return [
        {
            "candidate_ensembl_id": f"{TARGET}.9",
            "null_ensembl_id": f"ENSG{i:011d}.4",
            "path": PATH,
            "mode": "mask",
            "seed": 7,
            "rescue_excl_target": float(i),
        }
        for i in range(4, 4 + count)
    ]


def test_select_matched_nulls_uses_auditable_features_and_is_deterministic(tmp_path) -> None:
    cohort, null_ids = _synthetic_cohort()
    feature_table = pd.DataFrame(
        {
            "ensembl_id": [f"{TARGET}.7", f"{CANDIDATES[0]}.1", f"{CANDIDATES[1]}.2", f"{null_ids[0]}.1"],
            "log2fc": [1.5, 0.25, -0.5, -0.25],
        }
    )
    manifest = select_matched_nulls(
        cohort,
        f"{TARGET}.7",
        [f"{CANDIDATES[0]}.1", f"{CANDIDATES[1]}.2"],
        feature_table=feature_table,
        token_vocabulary=[TARGET, null_ids[0]],
    )

    assert manifest == select_matched_nulls(
        cohort,
        TARGET,
        CANDIDATES,
        feature_table=feature_table,
        token_vocabulary=[TARGET, null_ids[0]],
    )
    assert manifest["excluded_ids"] == sorted([TARGET, *CANDIDATES])
    selected_ids = {item["ensembl_id"] for item in manifest["selected_nulls"]}
    assert len(selected_ids) == 99
    assert selected_ids == set(null_ids)
    assert not selected_ids.intersection({TARGET, *CANDIDATES})

    target_features = manifest["target"]["features"]
    assert target_features["mean_expression"] == 1.0
    assert target_features["detection_rate"] == 0.5
    assert target_features["fc"] == 1.5
    assert target_features["token_rank"] == 1.0
    assert target_features["feature_order"] == ["mean_expression", "detection_rate", "fc", "token_rank"]
    null_features = next(
        item["match_features"] for item in manifest["selected_nulls"] if item["ensembl_id"] == null_ids[0]
    )
    assert null_features["fc"] == -0.25
    assert null_features["token_rank"] == 2.0
    assert "unavailable/default" not in manifest["feature_sources"]["fc"]

    output = write_null_selection_manifest(manifest, tmp_path / "selection.json")
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "perturbgen_null_selection/v1"


def test_select_matched_nulls_rejects_shortage_and_duplicate_ids() -> None:
    cohort, _ = _synthetic_cohort(null_count=98)
    with pytest.raises(ValueError, match="eligible nulls"):
        select_matched_nulls(cohort, TARGET, CANDIDATES)

    cohort, _ = _synthetic_cohort()
    with pytest.raises(ValueError, match="duplicate"):
        select_matched_nulls(cohort, TARGET, [CANDIDATES[0], f"{CANDIDATES[0]}.8"])
    with pytest.raises(ValueError, match="distinct"):
        select_matched_nulls(cohort, TARGET, [TARGET])
    with pytest.raises(ValueError, match="required_count.*99"):
        select_matched_nulls(cohort, TARGET, CANDIDATES, required_count=98)

    explicit_ids_missing = SimpleNamespace(X=cohort.X, var_names=cohort.var_names)
    with pytest.raises(ValueError, match=r"explicit var\['ensembl_id'\]"):
        select_matched_nulls(explicit_ids_missing, TARGET, CANDIDATES)


def test_summarize_null_distribution_validates_records_and_sorts_values(tmp_path) -> None:
    records = list(reversed(_records()))
    output_path = tmp_path / "distribution.json"
    payload = summarize_null_distribution(
        records,
        candidate_ensembl_id=TARGET,
        path=PATH,
        mode="mask",
        seed=7,
        output_path=output_path,
    )
    assert payload["schema_version"] == "perturbgen_null_distribution/v1"
    assert payload["null_ensembl_ids"] == sorted(payload["null_ensembl_ids"])
    assert payload["values"] == [float(i) for i in range(4, 103)]
    assert json.loads(output_path.read_text(encoding="utf-8"))["values"] == payload["values"]

    with pytest.raises(ValueError, match="at least"):
        summarize_null_distribution(_records(98), candidate_ensembl_id=TARGET, path=PATH, mode="mask", seed=7)
    nan_records = _records()
    nan_records[0]["rescue_excl_target"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        summarize_null_distribution(nan_records, candidate_ensembl_id=TARGET, path=PATH, mode="mask", seed=7)
    duplicate_records = _records()
    duplicate_records[1]["null_ensembl_id"] = duplicate_records[0]["null_ensembl_id"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_null_distribution(duplicate_records, candidate_ensembl_id=TARGET, path=PATH, mode="mask", seed=7)
    with pytest.raises(ValueError, match="candidate"):
        summarize_null_distribution(_records(), candidate_ensembl_id="ENSG00000000099", path=PATH, mode="mask", seed=7)
    with pytest.raises(ValueError, match="binding"):
        summarize_null_distribution(_records(), candidate_ensembl_id=TARGET, path=PATH, mode="pad", seed=7)


def test_load_null_distribution_manifest_supports_single_index_and_old_payloads(tmp_path) -> None:
    records = _records()
    single_path = tmp_path / "single.json"
    single = summarize_null_distribution(
        records,
        candidate_ensembl_id=TARGET,
        path=PATH,
        mode="mask",
        seed=7,
        output_path=single_path,
    )
    loaded = load_null_distribution_manifest(
        single_path,
        candidate_ensembl_id=TARGET,
        path_name=PATH,
        mode="mask",
        seed=7,
    )
    assert loaded["values"] == single["values"]

    second = dict(single)
    second.update({"path": "within_state", "mode": "pad", "seed": 8})
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"schema_version": "perturbgen_null_distribution/v1", "distributions": [single, second]}),
        encoding="utf-8",
    )
    indexed = load_null_distribution_manifest(
        index_path,
        candidate_ensembl_id=TARGET,
        path_name=PATH,
        mode="mask",
        seed=7,
    )
    assert indexed["values"] == single["values"]
    with pytest.raises(ValueError, match="binding|select exactly one"):
        load_null_distribution_manifest(
            index_path,
            candidate_ensembl_id=TARGET,
            path_name=PATH,
            mode="delete",
            seed=7,
        )

    old_list = tmp_path / "old-list.json"
    old_list.write_text(json.dumps([0.1] * 99), encoding="utf-8")
    assert load_null_distribution_manifest(old_list)["values"] == [0.1] * 99
    with pytest.raises(ValueError, match="unbound"):
        load_null_distribution_manifest(old_list, candidate_ensembl_id=TARGET)


def test_token_vocabulary_special_tokens_are_skipped_not_rejected() -> None:
    """Real embedding vocabularies ship <cls>/<pad>/... alongside ENSG keys.

    Special tokens carry no gene rank, so selection must ignore them instead
    of failing; ranked genes (including versioned ENSG) keep their ranks.
    """

    cohort, null_ids = _synthetic_cohort()
    vocabulary = {
        "<cls>": 0,
        "<eos>": 1,
        "<mask>": 2,
        "<pad>": 3,
        TARGET: 13453,
        f"{CANDIDATES[0]}.1": 9618,
        f"{CANDIDATES[1]}.2": 11423,
        null_ids[0]: 8218,
    }
    manifest = select_matched_nulls(
        cohort,
        TARGET,
        CANDIDATES,
        token_vocabulary=vocabulary,
    )
    assert manifest["feature_sources"]["token_rank"] == "token_vocabulary:mapping"
    assert manifest["target"]["features"]["token_rank"] == 13453.0
    features_by_gene = {item["ensembl_id"]: item["match_features"] for item in manifest["selected_nulls"]}
    assert features_by_gene[null_ids[0]]["token_rank"] == 8218.0
    assert features_by_gene[null_ids[1]]["token_rank"] == 0.0  # absent from vocabulary -> default

    sequence_manifest = select_matched_nulls(
        cohort,
        TARGET,
        CANDIDATES,
        token_vocabulary=["<cls>", TARGET, null_ids[0]],
    )
    sequence_features = {item["ensembl_id"]: item["match_features"] for item in sequence_manifest["selected_nulls"]}
    assert sequence_manifest["feature_sources"]["token_rank"] == "token_vocabulary:sequence; unknown/default=0.0"
    assert sequence_features[null_ids[0]]["token_rank"] == 3.0  # <cls> keeps its slot, genes rank by original position
    assert sequence_features[null_ids[1]]["token_rank"] == 0.0


@pytest.mark.parametrize("vocabulary", [{"<unk>": 4}, ["<unk>"]])
def test_token_vocabulary_rejects_unknown_special_tokens(vocabulary) -> None:
    cohort, _ = _synthetic_cohort()

    with pytest.raises(ValueError, match="invalid Ensembl gene id"):
        select_matched_nulls(cohort, TARGET, CANDIDATES, token_vocabulary=vocabulary)
