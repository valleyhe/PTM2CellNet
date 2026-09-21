"""Tests for signed-network loading and propagation (方案 §4.3/§5.3)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.ptm_research_config import PropagationConfig
from src.analysis.signed_network import (
    SignedNetworkContractError,
    load_signed_network,
    propagate_signed_scores,
    verify_network_release_binding,
)

EDGE_COLUMNS = (
    "source_id",
    "target_id",
    "edge_type",
    "effect_sign",
    "site",
    "species",
    "evidence",
    "confidence",
    "release",
)


def _edges_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {"species": "9606", "evidence": "test", "release": "2026-08"}
    base = {column: [] for column in EDGE_COLUMNS}
    for row in rows:
        for column in EDGE_COLUMNS:
            base[column].append(row.get(column, defaults.get(column, "")))
    return pd.DataFrame(base)


def _write(tmp_path, frame, name="network.tsv"):
    path = tmp_path / name
    frame.to_csv(path, sep="\t", index=False)
    return path


def _propagation(decay: float = 0.5, depth: int = 3, max_paths_per_seed: int | None = None) -> PropagationConfig:
    return PropagationConfig(
        max_depth=depth,
        decay=decay,
        gene_edge_types=("tf_regulation",),
        max_paths_per_seed=max_paths_per_seed,
    )


class TestLoadSignedNetwork:
    def test_valid_edges_load(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame(
                    [
                        {
                            "source_id": "GSK3B",
                            "target_id": "CDK5",
                            "edge_type": "protein_protein",
                            "effect_sign": "+1",
                            "confidence": 0.9,
                        },
                        {
                            "source_id": "CDK5",
                            "target_id": "MAPT",
                            "edge_type": "tf_regulation",
                            "effect_sign": "-1",
                            "confidence": 0.8,
                        },
                    ]
                ),
            )
        )
        assert network.audit.n_edges_propagation == 2
        assert network.audit.n_unsigned_rows == 0

    def test_unsigned_edges_are_excluded_from_propagation(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame(
                    [
                        {
                            "source_id": "A",
                            "target_id": "B",
                            "edge_type": "protein_protein",
                            "effect_sign": "",
                            "confidence": 0.9,
                        },
                        {
                            "source_id": "B",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 0.9,
                        },
                    ]
                ),
            )
        )
        assert network.audit.n_unsigned_rows == 1
        assert network.audit.n_edges_propagation == 1

    def test_self_loops_are_dropped_and_counted(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path, _edges_frame([{"source_id": "A", "target_id": "A", "edge_type": "x", "effect_sign": "+1"}])
            )
        )
        assert network.audit.n_self_loops == 1
        assert network.audit.n_edges_propagation == 0

    def test_sign_conflict_edges_are_dropped_not_picked(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame(
                    [
                        {
                            "source_id": "A",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 0.9,
                        },
                        {
                            "source_id": "A",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "-1",
                            "confidence": 0.7,
                        },
                        # A third same-sign row must not resurrect the conflicted pair.
                        {
                            "source_id": "A",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 0.6,
                        },
                    ]
                ),
            )
        )
        assert network.audit.n_sign_conflict_edges >= 1
        assert network.audit.n_edges_propagation == 0

    def test_parallel_same_sign_edges_keep_strongest_confidence(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame(
                    [
                        {
                            "source_id": "A",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 0.4,
                        },
                        {
                            "source_id": "A",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "1",
                            "confidence": 0.9,
                        },
                    ]
                ),
            )
        )
        assert network.audit.n_edges_propagation == 1
        assert network.adjacency["A"][0][2] == pytest.approx(0.9)

    def test_missing_confidence_defaults_to_one_and_is_counted(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame([{"source_id": "A", "target_id": "G", "edge_type": "tf_regulation", "effect_sign": "+1"}]),
            )
        )
        assert network.audit.n_default_confidence == 1
        assert network.adjacency["A"][0][2] == pytest.approx(1.0)

    def test_confidence_out_of_range_fails(self, tmp_path):
        with pytest.raises(SignedNetworkContractError, match="confidence"):
            load_signed_network(
                _write(
                    tmp_path,
                    _edges_frame(
                        [
                            {
                                "source_id": "A",
                                "target_id": "G",
                                "edge_type": "tf_regulation",
                                "effect_sign": "+1",
                                "confidence": 1.2,
                            }
                        ]
                    ),
                )
            )

    def test_non_numeric_confidence_fails_as_contract_error(self, tmp_path):
        with pytest.raises(SignedNetworkContractError, match="confidence"):
            load_signed_network(
                _write(
                    tmp_path,
                    _edges_frame(
                        [
                            {
                                "source_id": "A",
                                "target_id": "G",
                                "edge_type": "tf_regulation",
                                "effect_sign": "+1",
                                "confidence": "not-a-number",
                            }
                        ]
                    ),
                )
            )

    def test_missing_column_fails(self, tmp_path):
        frame = _edges_frame([{"source_id": "A", "target_id": "G", "edge_type": "x", "effect_sign": "+1"}])
        frame = frame.drop(columns=["release"])
        with pytest.raises(SignedNetworkContractError, match="release"):
            load_signed_network(_write(tmp_path, frame))

    def test_malformed_effect_sign_fails_instead_of_becoming_unsigned(self, tmp_path):
        with pytest.raises(SignedNetworkContractError, match="effect_sign"):
            load_signed_network(
                _write(
                    tmp_path,
                    _edges_frame(
                        [{"source_id": "A", "target_id": "G", "edge_type": "tf_regulation", "effect_sign": "0"}]
                    ),
                )
            )

    def test_expected_species_and_release_are_bound(self, tmp_path):
        path = _write(
            tmp_path,
            _edges_frame([{"source_id": "A", "target_id": "G", "edge_type": "tf_regulation", "effect_sign": "+1"}]),
        )
        with pytest.raises(SignedNetworkContractError, match="species"):
            load_signed_network(path, expected_species="10090")
        with pytest.raises(SignedNetworkContractError, match="releases"):
            load_signed_network(path, expected_release="2025-01")


class TestPropagateSignedScores:
    def _network(self, tmp_path):
        return load_signed_network(_write(tmp_path, _edges_frame(self._network_rows())))

    def test_signed_propagation_with_decay(self, tmp_path):
        network = self._network(tmp_path)
        result = propagate_signed_scores(network, {"KIN": 2.0}, config=_propagation(decay=0.5))
        by_target = {score.target_id: score for score in result.scores}
        # GENE_A receives KIN(+2) via TF1(+0.9, gene edge -0.8) and TF2(-0.5, gene edge +0.6):
        # 2 * 0.9 * -0.8 * 0.5^2 + 2 * -0.5 * 0.6 * 0.5^2 = -0.36 - 0.15 = -0.51
        assert by_target["GENE_A"].gene_score == pytest.approx(-0.51)
        assert by_target["GENE_A"].n_paths == 2
        assert by_target["GENE_A"].path_length_min == 2
        # GENE_B: KIN(+2) -> TF1(+0.9) -> GENE_B(+0.8): 2 * 0.9 * 0.8 * 0.5^2 = 0.36
        assert by_target["GENE_B"].gene_score == pytest.approx(0.36)
        assert by_target["GENE_B"].degree_normalized == pytest.approx(0.36)

    def test_inhibition_flips_downstream_sign(self, tmp_path):
        network = self._network(tmp_path)
        # A negative seed flips every contribution: both GENE_A paths become positive.
        result = propagate_signed_scores(network, {"KIN": -2.0}, config=_propagation(decay=0.5))
        by_target = {score.target_id: score for score in result.scores}
        assert by_target["GENE_A"].gene_score == pytest.approx(0.51)
        assert by_target["GENE_B"].gene_score == pytest.approx(-0.36)

    def test_multiple_paths_accumulate(self, tmp_path):
        network = self._network(tmp_path)
        # KIN seed contributes to GENE_A via both TF1 (2-hop) and TF2 (2-hop)
        result = propagate_signed_scores(network, {"KIN": 2.0}, config=_propagation(decay=1.0))
        by_target = {score.target_id: score for score in result.scores}
        assert by_target["GENE_A"].n_paths == 2
        assert by_target["GENE_A"].gene_score == pytest.approx(2 * 0.9 * -0.8 + 2 * -0.5 * 0.6)
        # seed total gene paths = 3 (GENE_A x2 + GENE_B x1)
        assert by_target["GENE_A"].network_coverage == pytest.approx(2 / 3)
        assert by_target["GENE_B"].network_coverage == pytest.approx(1 / 3)
        assert by_target["GENE_A"].degree_normalized == pytest.approx((2 * 0.9 * -0.8 + 2 * -0.5 * 0.6) / 2)
        assert result.diagnostics["per_seed_path_counts"] == {"KIN": 3}
        assert result.diagnostics["path_count_distribution"] == {
            "n_seeds": 1,
            "min": 3,
            "max": 3,
            "mean": 3.0,
        }

    def test_max_paths_per_seed_fails_before_truncating_paths(self, tmp_path):
        network = self._network(tmp_path)
        with pytest.raises(SignedNetworkContractError, match="max_paths_per_seed.*lower max_depth.*network filtering"):
            propagate_signed_scores(network, {"KIN": 2.0}, config=_propagation(max_paths_per_seed=2))

    def test_none_max_paths_per_seed_preserves_default_result(self, tmp_path):
        network = self._network(tmp_path)
        default = propagate_signed_scores(network, {"KIN": 2.0}, config=_propagation())
        explicit_none = propagate_signed_scores(
            network,
            {"KIN": 2.0},
            config=_propagation(max_paths_per_seed=None),
        )
        assert explicit_none == default

    def test_gene_edge_terminates_path(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame(
                    [
                        {
                            "source_id": "TF",
                            "target_id": "GENE_A",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 1.0,
                        },
                        {
                            "source_id": "GENE_A",
                            "target_id": "GENE_B",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 1.0,
                        },
                    ]
                ),
            )
        )
        result = propagate_signed_scores(network, {"TF": 1.0}, config=_propagation(decay=1.0))
        targets = {score.target_id for score in result.scores}
        assert targets == {"GENE_A"}
        assert all(score.path_length_min == 1 for score in result.scores)

    def test_depth_limit_bounds_reach(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame(
                    [
                        {
                            "source_id": "A",
                            "target_id": "B",
                            "edge_type": "protein_protein",
                            "effect_sign": "+1",
                            "confidence": 1.0,
                        },
                        {
                            "source_id": "B",
                            "target_id": "C",
                            "edge_type": "protein_protein",
                            "effect_sign": "+1",
                            "confidence": 1.0,
                        },
                        {
                            "source_id": "C",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 1.0,
                        },
                    ]
                ),
            )
        )
        shallow = propagate_signed_scores(network, {"A": 1.0}, config=_propagation(depth=2))
        assert list(shallow.scores) == []
        deep = propagate_signed_scores(network, {"A": 1.0}, config=_propagation(depth=3))
        assert {score.target_id for score in deep.scores} == {"G"}
        assert deep.scores[0].path_length_min == 3

    def test_seed_without_node_is_reported(self, tmp_path):
        network = self._network(tmp_path)
        result = propagate_signed_scores(network, {"KIN": 1.0, "GHOST": 1.0}, config=_propagation())
        assert result.seeds_without_node == ("GHOST",)
        assert result.seeds_matched == ("KIN",)

    def test_zero_seed_activity_fails(self, tmp_path):
        network = self._network(tmp_path)
        with pytest.raises(SignedNetworkContractError, match="non-zero"):
            propagate_signed_scores(network, {"KIN": 0.0}, config=_propagation())

    def test_opposing_paths_can_cancel_to_indeterminate(self, tmp_path):
        network = load_signed_network(
            _write(
                tmp_path,
                _edges_frame(
                    [
                        {
                            "source_id": "A",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "effect_sign": "+1",
                            "confidence": 1.0,
                        },
                        {
                            "source_id": "A",
                            "target_id": "G",
                            "edge_type": "tf_regulation",
                            "site": "s2",
                            "effect_sign": "-1",
                            "confidence": 1.0,
                        },
                    ]
                ),
            )
        )
        # Different sites keep both parallel edges (no conflict), and they cancel.
        result = propagate_signed_scores(network, {"A": 1.0}, config=_propagation(decay=1.0))
        assert result.scores[0].gene_score == pytest.approx(0.0)
        assert result.scores[0].n_paths == 2

    def test_cli_manifest_records_serializable_path_limit_metadata(self, tmp_path):
        from scripts.build_ptm_global_gene_scores import main

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "schema_version: ptm2cellnet.ptm-research-config/v1",
                    "research_objective: association",
                    "reference_axis: disease_minus_normal",
                    'contrast: "disease-minus-normal"',
                    "primary_activity_method: KSTAR",
                    'network_release: "2026-08"',
                    "cell_types: [EX]",
                    "cohort_h5ad: unused.h5ad",
                    "cohort_pairing: between_donor",
                    'species: "9606"',
                    "ptm_cohort: CPTAC_TEST",
                    "deg_max_fdr: 0.05",
                    "min_donors_per_state: 3",
                    "replicate_policy: mean",
                    "propagation:",
                    "  max_depth: 3",
                    "  decay: 0.5",
                    "  gene_edge_types: [tf_regulation]",
                    "  max_paths_per_seed: 3",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        activity_path = tmp_path / "activity.tsv"
        activity_path.write_text(
            "\n".join(
                [
                    "activity_unit\tregulator_id\tregulator_type\tcondition_or_contrast\tactivity_score\t"
                    "activity_direction\tactivity_pvalue\tactivity_qvalue\tn_substrates\tnetwork_coverage\t"
                    "method\tmethod_version\tinput_manifest",
                    "zscore\tKIN\tkinase\tdisease-minus-normal\t2.0\tup\t0.01\t0.05\t3\t1.0\tKSTAR\t1.0\tinput.json",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        network_path = _write(tmp_path, _edges_frame(self._network_rows()))
        id_map_path = tmp_path / "network_id_map.tsv"
        id_map_path.write_text(
            "network_id\tgene_symbol\tensembl_id\nGENE_A\tGENEA\tENSG00000000001\n",
            encoding="utf-8",
        )
        output_path = tmp_path / "scores.tsv"
        manifest_path = tmp_path / "manifest.json"

        assert (
            main(
                [
                    "--config",
                    str(config_path),
                    "--activity-tsv",
                    str(activity_path),
                    "--network-tsv",
                    str(network_path),
                    "--network-id-map",
                    str(id_map_path),
                    "--output-tsv",
                    str(output_path),
                    "--manifest-output",
                    str(manifest_path),
                ]
            )
            == 0
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        propagation = manifest["propagation"]
        assert propagation["max_paths_per_seed"] == 3
        assert propagation["per_seed_path_counts"] == {"KIN": 3}
        assert propagation["path_count_distribution"] == {
            "n_seeds": 1,
            "min": 3,
            "max": 3,
            "mean": 3.0,
        }
        json.dumps(manifest)

    @staticmethod
    def _network_rows():
        return [
            {
                "source_id": "KIN",
                "target_id": "TF1",
                "edge_type": "protein_protein",
                "effect_sign": "+1",
                "confidence": 0.9,
            },
            {
                "source_id": "TF1",
                "target_id": "GENE_A",
                "edge_type": "tf_regulation",
                "effect_sign": "-1",
                "confidence": 0.8,
            },
            {
                "source_id": "TF1",
                "target_id": "GENE_B",
                "edge_type": "tf_regulation",
                "effect_sign": "+1",
                "confidence": 0.8,
            },
            {
                "source_id": "KIN",
                "target_id": "TF2",
                "edge_type": "protein_protein",
                "effect_sign": "-1",
                "confidence": 0.5,
            },
            {
                "source_id": "TF2",
                "target_id": "GENE_A",
                "edge_type": "tf_regulation",
                "effect_sign": "+1",
                "confidence": 0.6,
            },
        ]


def _release_binding_fixture(tmp_path, *, rows: int = 3, sha: str | None = None, release: str = "test-release-2026"):
    import hashlib

    header = "\t".join(EDGE_COLUMNS) + "\n"
    body = "".join(
        f"KIN{index}\tGENE_{index}\tkinase_substrate:signaling\t+1\t\t9606\ttest\t0.8\t{release}\n"
        for index in range(rows)
    )
    network = tmp_path / "network.tsv"
    network.write_text(header + body, encoding="utf-8")
    digest = sha or hashlib.sha256(network.read_bytes()).hexdigest()
    manifest = tmp_path / "release_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "ptm2cellnet.signed-network-release/v1",
                "release": release,
                "combined": {"path": str(network), "rows": rows, "sha256": digest},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return network, manifest, release


def test_network_release_binding_passes_on_three_way_match(tmp_path):
    network, manifest, release = _release_binding_fixture(tmp_path)
    result = verify_network_release_binding(network, manifest, expected_release=release)
    assert result["release"] == release
    assert result["rows"] == 3


def test_network_release_binding_rejects_drift(tmp_path):
    network, manifest, release = _release_binding_fixture(tmp_path)
    # config text drift
    with pytest.raises(SignedNetworkContractError, match="frozen config binds"):
        verify_network_release_binding(network, manifest, expected_release="other-release")
    # asset drift: rewrite the network without refreshing the manifest hash
    drifted = tmp_path / "drifted.tsv"
    drifted.write_text(network.read_text(encoding="utf-8") + network.read_text(encoding="utf-8").splitlines()[1] + "\n")
    with pytest.raises(SignedNetworkContractError, match="drifted"):
        verify_network_release_binding(drifted, manifest, expected_release=release)
    # manifest records a different row count
    bad_rows_manifest = tmp_path / "bad_rows.json"
    import hashlib

    bad_rows_manifest.write_text(
        json.dumps(
            {
                "schema_version": "ptm2cellnet.signed-network-release/v1",
                "release": release,
                "combined": {
                    "path": str(network),
                    "rows": 999,
                    "sha256": hashlib.sha256(network.read_bytes()).hexdigest(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SignedNetworkContractError, match="rows"):
        verify_network_release_binding(network, bad_rows_manifest, expected_release=release)
