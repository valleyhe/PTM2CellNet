"""Signed network loading and signed-score propagation (方案 §4.3/§5.3).

The network table is an external frozen asset (OmniPath signed
signaling/TF-regulin export). This module only:

* validates the §4.3 edge contract (``source_id, target_id, edge_type,
  effect_sign, site, species, evidence, confidence, release``);
* keeps unsigned edges out of formal propagation (they may only feed
  unsigned coverage statistics — 方案 §4.3);
* resolves parallel edges: same-sign parallel edges collapse to the
  strongest confidence, sign conflicts are removed from propagation and
  counted, never silently picked;
* propagates signed regulator activities through bounded simple paths
  (方案 §5.3): a path terminates at a gene when its last hop uses one of the
  frozen ``gene_edge_types``; each path contributes
  ``activity × Π(sign_e × confidence_e) × decay^length``.

The hardcoded pathways/output genes/fixed effect factors in
``src/models/signaling_network.py`` are NOT used here (方案 §5.3 explicitly
excludes them from formal propagation).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, TypeAlias

import pandas as pd

from src.analysis.ptm_research_config import PropagationConfig

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

SIGNED_NETWORK_REQUIRED_COLUMNS: tuple[str, ...] = (
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

#: Parallel-edge key: the same (source, target, edge_type, site) tuple is one
#: biological relationship; multiple evidences of it collapse, sign conflicts drop.
_PARALLEL_KEY_COLUMNS = ("source_id", "target_id", "edge_type", "site")


class SignedNetworkContractError(ValueError):
    """Raised when a signed-network table violates the plan contract."""


@dataclass(frozen=True)
class SignedNetworkAudit:
    """Audit record of edge loading and parallel-edge resolution."""

    n_rows_input: int
    n_unsigned_rows: int
    n_self_loops: int
    n_sign_conflict_edges: int
    n_default_confidence: int
    n_edges_propagation: int
    species: tuple[str, ...]
    releases: tuple[str, ...]
    edge_types: tuple[str, ...]


@dataclass(frozen=True)
class NetworkEdge:
    source_id: str
    target_id: str
    edge_type: str
    effect_sign: int
    confidence: float
    site: str = ""
    evidence: str = ""
    release: str = ""


@dataclass
class SignedNetwork:
    """Signed adjacency with audit statistics."""

    adjacency: dict[str, list[tuple[str, int, float, str]]]
    in_degree: dict[str, int]
    audit: SignedNetworkAudit

    def has_node(self, node_id: str) -> bool:
        return node_id in self.adjacency or self.in_degree.get(node_id, 0) > 0


@dataclass(frozen=True)
class TargetPropagationScore:
    """Aggregated signed score for one (seed regulator, target gene) pair."""

    source_id: str
    target_id: str
    gene_score: float
    n_paths: int
    path_length_min: int
    #: Fraction of the seed's gene-terminating paths that reach this target.
    network_coverage: float
    #: Mean signed per-path contribution; guards degree inflation (方案 §5.3).
    degree_normalized: float


@dataclass(frozen=True)
class PropagationResult:
    scores: tuple[TargetPropagationScore, ...]
    seeds_matched: tuple[str, ...]
    seeds_without_node: tuple[str, ...]
    n_gene_paths_total: int = 0
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)


def load_signed_network(
    path: str | Path,
    *,
    expected_species: str | None = None,
    expected_release: str | None = None,
) -> SignedNetwork:
    """Load and validate the signed edge table (方案 §4.3)."""

    resolved = Path(path).expanduser().resolve(strict=True)
    frame = pd.read_csv(resolved, sep="\t", dtype={"effect_sign": "string", "confidence": "string", "site": "string"})
    missing = [column for column in SIGNED_NETWORK_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SignedNetworkContractError(f"signed network is missing required columns: {', '.join(missing)}")
    for column in ("source_id", "target_id", "edge_type", "species", "evidence", "release"):
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise SignedNetworkContractError(f"signed network column {column!r} must not contain empty values")
    species = tuple(sorted(set(frame["species"].astype(str).str.strip())))
    releases = tuple(sorted(set(frame["release"].astype(str).str.strip())))
    if expected_species is not None:
        expected = str(expected_species).strip()
        if not expected or species != (expected,):
            raise SignedNetworkContractError(
                f"signed network species {species} does not match frozen species {expected!r}"
            )
    if expected_release is not None:
        expected = str(expected_release).strip()
        if not expected or releases != (expected,):
            raise SignedNetworkContractError(
                f"signed network releases {releases} do not match frozen release {expected!r}"
            )
    n_input = len(frame)
    n_self_loops = 0
    n_unsigned = 0
    n_default_confidence = 0
    n_conflicts = 0
    parallel: dict[tuple[str, str, str, str], NetworkEdge] = {}
    conflicted: set[tuple[str, str, str, str]] = set()
    edge_types: set[str] = set()
    for record in frame.to_dict("records"):
        source_id = str(record["source_id"]).strip()
        target_id = str(record["target_id"]).strip()
        edge_type = str(record["edge_type"]).strip()
        site = "" if pd.isna(record["site"]) else str(record["site"]).strip()
        if source_id == target_id:
            n_self_loops += 1
            continue
        raw_sign = record["effect_sign"]
        if raw_sign is None or pd.isna(raw_sign) or str(raw_sign).strip() == "":
            n_unsigned += 1
            continue
        effect_sign = _parse_effect_sign(raw_sign)
        if effect_sign is None:
            raise SignedNetworkContractError(
                f"effect_sign must be a signed +1/-1 value or empty; got {raw_sign!r} on {source_id}->{target_id}"
            )
        raw_confidence = record["confidence"]
        if raw_confidence is None or pd.isna(raw_confidence) or str(raw_confidence).strip() == "":
            confidence = 1.0
            n_default_confidence += 1
        else:
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError) as exc:
                raise SignedNetworkContractError(
                    f"confidence must be numeric where present; got {raw_confidence!r} on {source_id}->{target_id}"
                ) from exc
            if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
                raise SignedNetworkContractError(
                    f"confidence must be within (0, 1] where present; got {raw_confidence!r} on {source_id}->{target_id}"
                )
        edge_types.add(edge_type)
        key = (source_id, target_id, edge_type, site)
        edge = NetworkEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            effect_sign=effect_sign,
            confidence=confidence,
            site=site,
            evidence=str(record["evidence"]).strip(),
            release=str(record["release"]).strip(),
        )
        previous = parallel.get(key)
        if key in conflicted:
            continue
        if previous is None:
            parallel[key] = edge
        elif previous.effect_sign != edge.effect_sign:
            n_conflicts += 1
            # Sign conflicts are dropped from propagation entirely; a silently
            # chosen winner would fabricate a direction (方案 §4.3).
            del parallel[key]
            conflicted.add(key)
        elif edge.confidence > previous.confidence:
            parallel[key] = edge
    adjacency: dict[str, list[tuple[str, int, float, str]]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    for edge in parallel.values():
        adjacency[edge.source_id].append((edge.target_id, edge.effect_sign, edge.confidence, edge.edge_type))
        in_degree[edge.target_id] += 1
    audit = SignedNetworkAudit(
        n_rows_input=n_input,
        n_unsigned_rows=n_unsigned,
        n_self_loops=n_self_loops,
        n_sign_conflict_edges=n_conflicts,
        n_default_confidence=n_default_confidence,
        n_edges_propagation=len(parallel),
        species=species,
        releases=releases,
        edge_types=tuple(sorted(edge_types)),
    )
    return SignedNetwork(adjacency=dict(adjacency), in_degree=dict(in_degree), audit=audit)


def _parse_effect_sign(raw: object) -> int | None:
    text = str(raw).strip()
    if text in {"+1", "+1.0", "1", "1.0", "+", "activation"}:
        return 1
    if text in {"-1", "-1.0", "-", "inhibition", "repression"}:
        return -1
    return None


def propagate_signed_scores(
    network: SignedNetwork,
    activities: Mapping[str, float],
    *,
    config: PropagationConfig,
) -> PropagationResult:
    """Propagate signed seed activities to gene targets (方案 §5.3).

    Deterministic bounded simple-path enumeration. A path accumulates
    ``sign_e × confidence_e`` per hop and ``decay^length`` overall; it
    terminates as a scored gene path when its last hop uses one of the frozen
    ``gene_edge_types``. Scores are summed per (seed, target) pair.
    """

    gene_edge_types = frozenset(config.gene_edge_types)
    per_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_pair_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    seed_path_counts: dict[str, int] = {}
    seeds_matched: list[str] = []
    seeds_without_node: list[str] = []
    n_gene_paths_total = 0

    for seed_id in sorted(activities):
        activity = float(activities[seed_id])
        if not math.isfinite(activity) or activity == 0.0:
            raise SignedNetworkContractError(
                f"seed activity for {seed_id!r} must be finite and non-zero; directionless seeds cannot propagate"
            )
        if not network.has_node(seed_id):
            seed_path_counts[seed_id] = 0
            seeds_without_node.append(seed_id)
            continue
        seeds_matched.append(seed_id)
        seed_gene_paths = _enumerate_seed_paths(
            network.adjacency,
            seed_id,
            max_depth=config.max_depth,
            gene_edge_types=gene_edge_types,
            max_paths_per_seed=config.max_paths_per_seed,
        )
        seed_path_counts[seed_id] = sum(len(contributions) for contributions, _ in seed_gene_paths.values())
        for target_id, (contributions, lengths) in seed_gene_paths.items():
            n_gene_paths_total += len(contributions)
            pair = (seed_id, target_id)
            per_pair[pair].extend(
                activity * contribution * (config.decay**length)
                for contribution, length in zip(contributions, lengths, strict=True)
            )
            per_pair_lengths[pair].extend(lengths)
    scores: list[TargetPropagationScore] = []
    for (seed_id, target_id), contributions in sorted(per_pair.items()):
        gene_score = math.fsum(contributions)
        lengths = per_pair_lengths[(seed_id, target_id)]
        seed_total = seed_path_counts[seed_id]
        scores.append(
            TargetPropagationScore(
                source_id=seed_id,
                target_id=target_id,
                gene_score=gene_score,
                n_paths=len(contributions),
                path_length_min=min(lengths),
                network_coverage=len(contributions) / seed_total if seed_total else 0.0,
                degree_normalized=gene_score / len(contributions),
            )
        )
    path_counts = tuple(seed_path_counts.values())
    return PropagationResult(
        scores=tuple(scores),
        seeds_matched=tuple(seeds_matched),
        seeds_without_node=tuple(seeds_without_node),
        n_gene_paths_total=n_gene_paths_total,
        diagnostics={
            "per_seed_path_counts": dict(sorted(seed_path_counts.items())),
            "path_count_distribution": {
                "n_seeds": len(path_counts),
                "min": min(path_counts, default=0),
                "max": max(path_counts, default=0),
                "mean": math.fsum(path_counts) / len(path_counts) if path_counts else 0.0,
            },
        },
    )


def _enumerate_seed_paths(
    adjacency: Mapping[str, list[tuple[str, int, float, str]]],
    seed_id: str,
    *,
    max_depth: int,
    gene_edge_types: frozenset[str],
    max_paths_per_seed: int | None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Enumerate gene-terminating simple paths from one seed.

    Returns per target the list of signed path factors ``Π(sign_e ×
    confidence_e)`` (decay applied later per length) and the path lengths.
    A path stops either at a gene-edge hop (scored gene path) or when it
    cannot be extended within ``max_depth``.
    """

    results: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    adjacency = dict(adjacency)
    # Iterative DFS: stack of (node, visited, accumulated factor, depth).
    stack: list[tuple[str, frozenset[str], float, int]] = [(seed_id, frozenset({seed_id}), 1.0, 0)]
    n_gene_paths = 0
    while stack:
        node, visited, factor, depth = stack.pop()
        for target_id, effect_sign, confidence, edge_type in sorted(
            adjacency.get(node, ()), key=lambda item: (item[0], item[3])
        ):
            if target_id in visited:
                continue
            hop = factor * (effect_sign * confidence)
            if edge_type in gene_edge_types:
                n_gene_paths += 1
                if max_paths_per_seed is not None and n_gene_paths > max_paths_per_seed:
                    raise SignedNetworkContractError(
                        f"max_paths_per_seed={max_paths_per_seed} exceeded for seed {seed_id!r} "
                        f"after {n_gene_paths} gene-terminating simple paths; lower max_depth or "
                        "tighten network filtering"
                    )
                contributions, lengths = results[target_id]
                contributions.append(hop)
                lengths.append(depth + 1)
                continue
            if depth + 1 < max_depth:
                stack.append((target_id, visited | {target_id}, hop, depth + 1))
    return dict(results)


def iter_seed_target_pairs(result: PropagationResult) -> Iterable[tuple[str, str]]:
    for score in result.scores:
        yield score.source_id, score.target_id


__all__ = [
    "NetworkEdge",
    "PropagationResult",
    "SignedNetwork",
    "SignedNetworkAudit",
    "SignedNetworkContractError",
    "SIGNED_NETWORK_REQUIRED_COLUMNS",
    "TargetPropagationScore",
    "load_signed_network",
    "propagate_signed_scores",
]
