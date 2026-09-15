#!/usr/bin/env python3
"""Import the OmniPath kinase-substrate snapshot into PTM2CellNet.

Consumes ``data/raw/kinase_substrate/omnipath_enzsub.tsv`` (OmniPath
``enzsub`` endpoint, 2026-08 snapshot, 41,506 records, UniProt IDs) and
produces the *canonical signaling-edge* artifacts that downstream
consumers (``src/analysis/pathway_integration.py``, the cross-scale
NPZ assembler) expect:

* ``kinase_substrate_edges.tsv`` — canonical edge table with columns
  ``source_gene``, ``target_gene``, ``edge_type``, ``source_site``,
  ``evidence``, ``species``, ``release`` (gene symbols resolved from
  UniProt accessions; unmapped IDs are kept verbatim, never dropped);
* ``uniprot_to_gene.tsv`` — the UniProt -> gene symbol mapping cache;
* ``import_manifest.json`` — provenance (source sha256, parsing stats,
  mapping rate, vocabulary version).

UniProt -> gene symbol resolution (this closes the loader breakage
registered in ``data/manifests/datasets.yaml`` for ``kinase_substrate``,
whose previous ``loader`` pointed at ``scripts/import_norman_adamson.py``
without any OmniPath logic):

1. A local mapping cache (``--mapping-file``, or the cache written by a
   previous run in the output directory) is used when available —
   fully offline;
2. Otherwise ``--online-mapping`` queries the UniProt REST ID-mapping
   service (``UniProtKB_AC-ID -> Gene_Name``) in batches and caches the
   result for subsequent offline runs.

Design notes
------------
* Pure stdlib — no pandas/requests hard dependency (``requests`` is
  imported lazily only for the opt-in online mapping path).
* Fail-fast for missing inputs; unknown mapping results degrade to
  keeping the accession and are counted in the manifest.
* The edge table schema matches the ``canonical_schema`` declared in
  ``data/manifests/datasets.yaml`` for ``kinase_substrate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENZSUB_FILE = "omnipath_enzsub.tsv"
DEFAULT_RAW_ROOT = "data/raw/kinase_substrate"
DEFAULT_OUTPUT = "data/processed/kinase_substrate"
MAPPING_CACHE_NAME = "uniprot_to_gene.tsv"
EDGES_NAME = "kinase_substrate_edges.tsv"
MANIFEST_NAME = "import_manifest.json"

#: UniProt REST ID-mapping service (only used with ``--online-mapping``).
UNIPROT_IDMAPPING_URL = "https://rest.uniprot.org/idmapping"

#: Batch size for the UniProt ID-mapping submissions (service limit ~100).
ONLINE_BATCH_SIZE = 100
#: Total polling budget (seconds) across all submitted jobs.
ONLINE_POLL_BUDGET_SECONDS = 300
#: Polling interval between rounds.
ONLINE_POLL_INTERVAL_SECONDS = 2.0

#: Species label recorded for this snapshot. The 2026-08 enzsub snapshot
#: was exported with the OmniPath default organism filter (human, 9606);
#: spot checks of the accessions (e.g. P06239 = LCK) confirm human
#: entries. Recorded here so ``species_recorded`` stays truthful.
SPECIES = "9606"
#: Snapshot release label (matches datasets.yaml reference).
RELEASE = "2026-08"

#: Evidence provenance string (aggregated sources, per OmniPath docs).
EVIDENCE = "OmniPath enzsub (aggregated; PhosphoSitePlus/KEA/ProtMapper etc.)"

EDGE_TABLE_COLUMNS = (
    "source_gene",
    "target_gene",
    "edge_type",
    "source_site",
    "evidence",
    "species",
    "release",
)


class ImportError_(RuntimeError):
    """Raised when the required OmniPath input is missing or malformed."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_enzsub(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    """Parse the OmniPath enzsub TSV into edge records.

    Expected columns: ``enzyme``, ``substrate``, ``residue_type``,
    ``residue_offset``, ``modification``. Rows with fewer columns are
    rejected with a descriptive error (fail-fast, not silent skip).
    """
    if not Path(path).is_file():
        raise ImportError_(f"缺少输入文件: {path}")
    records: List[Dict[str, str]] = []
    with open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cells = [cell.strip() for cell in line.split("\t")]
            if line_no == 1:
                if cells[:5] != ["enzyme", "substrate", "residue_type", "residue_offset", "modification"]:
                    raise ImportError_(f"输入文件列头不符合 enzsub 契约: {path}:1 -> {cells[:5]!r}")
                continue
            if len(cells) < 5:
                raise ImportError_(f"enzsub 行列数不足（期望 >=5，实际 {len(cells)}）: {path}:{line_no}")
            records.append(
                {
                    "enzyme": cells[0],
                    "substrate": cells[1],
                    "residue_type": cells[2],
                    "residue_offset": cells[3],
                    "modification": cells[4],
                }
            )
            if max_rows is not None and len(records) >= max_rows:
                break
    if not records:
        raise ImportError_(f"enzsub 文件无数据行: {path}")
    return records


def collect_uniprot_ids(records: Sequence[Mapping[str, str]]) -> List[str]:
    """All unique UniProt accessions across enzyme/substrate columns."""
    ids: set = set()
    for record in records:
        ids.add(record["enzyme"])
        ids.add(record["substrate"])
    return sorted(ids)


# ---------------------------------------------------------------------------
# UniProt -> gene symbol resolution
# ---------------------------------------------------------------------------


def load_mapping_file(path: Path) -> Dict[str, str]:
    """Load a two-column TSV mapping (uniprot_id -> gene_symbol).

    Returns an empty mapping when the file does not exist (callers treat
    that as "no local cache available").
    """
    mapping: Dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return mapping
    with open(p, "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            cells = [cell.strip() for cell in line.split("\t")]
            if len(cells) < 2 or not cells[0] or not cells[1]:
                raise ImportError_(f"映射文件行格式错误（期望 uniprot_id<TAB>gene_symbol）: {path}:{line_no}")
            mapping[cells[0].upper()] = cells[1]
    return mapping


def write_mapping_file(path: Path, mapping: Mapping[str, str]) -> None:
    """Persist a UniProt -> gene mapping as a two-column TSV cache."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# uniprot_id\tgene_symbol\n")
        for uniprot_id in sorted(mapping):
            handle.write(f"{uniprot_id}\t{mapping[uniprot_id]}\n")


def resolve_online(uniprot_ids: Sequence[str], *, session: Optional[Any] = None) -> Dict[str, str]:
    """Query the UniProt REST ID-mapping service (UniProtKB_AC-ID -> Gene_Name).

    All batches are submitted first, then polled together (the service
    processes jobs asynchronously; per-batch submit+wait serialises on
    nothing and wastes wall-clock time). Only the IDs that actually
    resolve are returned; unresolvable IDs are simply absent from the
    result (callers keep the accession).
    """
    import requests

    if not uniprot_ids:
        return {}
    s = session if session is not None else requests
    mapping: Dict[str, str] = {}
    jobs: List[str] = []
    finished_jobs: Dict[str, List[str]] = {}  # job id -> batch ids (for /results fallback)
    import time

    # Phase 1: submit every batch.
    for start in range(0, len(uniprot_ids), ONLINE_BATCH_SIZE):
        batch = uniprot_ids[start : start + ONLINE_BATCH_SIZE]
        try:
            submit_response = s.post(
                f"{UNIPROT_IDMAPPING_URL}/run",
                data={"from": "UniProtKB_AC-ID", "to": "Gene_Name", "ids": ",".join(batch)},
                timeout=30,
            )
            submit_response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - network failures degrade to no mapping
            raise ImportError_(f"UniProt ID-mapping 提交失败（批次 {start}:{start + len(batch)}）: {exc}") from exc
        job_id = _extract_job_id(submit_response)
        if not job_id:
            raise ImportError_("UniProt ID-mapping 返回空 job id")
        jobs.append(job_id)
        finished_jobs[job_id] = list(batch)

    # Phase 2: poll all jobs until finished or the global budget runs out.
    # The /status endpoint reports RUNNING while a job is queued/processing
    # and switches to an inline {"results": [...], "failedIds": [...]}
    # payload (no jobStatus key) once finished; some flavours keep
    # jobStatus in {FINISHED, COMPLETED, FAILED}. NOTE: the inline results
    # are truncated to 25 entries — the full result set must come from
    # /results/{job} (Phase 3).
    deadline = time.monotonic() + ONLINE_POLL_BUDGET_SECONDS
    pending = set(jobs)
    while pending and time.monotonic() < deadline:
        for job_id in list(pending):
            try:
                details_response = s.get(f"{UNIPROT_IDMAPPING_URL}/status/{job_id}", timeout=30)
            except Exception:  # noqa: BLE001
                continue
            if details_response.status_code != 200:
                continue
            try:
                payload = details_response.json()
            except ValueError:
                continue
            if "results" in payload or payload.get("jobStatus") in ("FINISHED", "COMPLETED", "FAILED"):
                pending.discard(job_id)
        if pending:
            time.sleep(ONLINE_POLL_INTERVAL_SECONDS)

    if pending:
        raise ImportError_(f"UniProt ID-mapping jobs {sorted(pending)} 未在预算时间内完成")

    # Phase 3: fetch the full result set for every job from /results
    # (the inline /status payload is truncated, so it is never trusted).
    for job_id, _batch_ids in finished_jobs.items():
        try:
            results_response = s.get(
                f"{UNIPROT_IDMAPPING_URL}/results/{job_id}",
                params={"size": 500},
                timeout=30,
            )
            results_response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise ImportError_(f"UniProt ID-mapping 结果获取失败: {exc}") from exc
        for entry in results_response.json().get("results", []):
            from_id = str(entry.get("from", "")).upper()
            to_id = entry.get("to")
            gene = to_id.get("primaryAccession") if isinstance(to_id, dict) else to_id
            if from_id and gene:
                mapping[from_id] = str(gene)
    return mapping


def _extract_job_id(submit_response: Any) -> str:
    """Parse the submission response; prefers the JSON ``jobId`` field
    and falls back to a plain-text job id for older API flavours."""
    try:
        payload = submit_response.json()
        if isinstance(payload, dict):
            return str(payload.get("jobId", "")).strip()
    except ValueError:
        pass
    return submit_response.text.strip()


def resolve_genes(
    uniprot_ids: Sequence[str],
    *,
    mapping_file: Optional[Path] = None,
    online: bool = False,
    session: Optional[Any] = None,
) -> Dict[str, str]:
    """Resolve UniProt accessions to gene symbols.

    Priority: local cache file > (opt-in) online service. The merged
    result contains an entry only for IDs that resolved; callers must
    keep unresolved accessions verbatim.
    """
    resolved: Dict[str, str] = {}
    if mapping_file is not None:
        resolved.update(load_mapping_file(mapping_file))
    missing = [uid for uid in uniprot_ids if uid.upper() not in resolved]
    if online and missing:
        online_resolved = resolve_online(missing, session=session)
        resolved.update(online_resolved)
    return resolved


# ---------------------------------------------------------------------------
# Canonical edge table
# ---------------------------------------------------------------------------


def build_edge_table(
    records: Sequence[Mapping[str, str]],
    mapping: Mapping[str, str],
) -> List[Dict[str, str]]:
    """Map enzsub records to canonical edges.

    Unresolved accessions are kept verbatim in source_gene/target_gene
    (never dropped); the manifest reports the unmapped fraction.
    """
    edges: List[Dict[str, str]] = []
    for record in records:
        source = mapping.get(record["enzyme"].upper(), record["enzyme"])
        target = mapping.get(record["substrate"].upper(), record["substrate"])
        residue = record.get("residue_type", "")
        offset = record.get("residue_offset", "")
        edges.append(
            {
                "source_gene": source,
                "target_gene": target,
                "edge_type": f"kinase_substrate:{record.get('modification', '')}",
                "source_site": f"{residue}{offset}" if residue else "",
                "evidence": EVIDENCE,
                "species": SPECIES,
                "release": RELEASE,
            }
        )
    return edges


def write_edge_table(path: Path, edges: Sequence[Mapping[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(EDGE_TABLE_COLUMNS) + "\n")
        for edge in edges:
            handle.write("\t".join(str(edge[column]) for column in EDGE_TABLE_COLUMNS) + "\n")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_import_manifest(
    output_dir: Path,
    *,
    source_path: Path,
    source_sha256: str,
    n_records: int,
    n_unique_ids: int,
    n_mapped_ids: int,
    n_edges: int,
    edge_type_counts: Mapping[str, int],
    args: argparse.Namespace,
) -> Path:
    manifest = {
        "schema_version": "ptm2cellnet.kinase-substrate.import.v1",
        "source": {
            "file": str(source_path),
            "sha256": source_sha256,
            "reference": "OmniPath enzsub endpoint (2026-08 snapshot), 41506 records",
        },
        "summary": {
            "n_records": n_records,
            "n_unique_uniprot_ids": n_unique_ids,
            "n_mapped_uniprot_ids": n_mapped_ids,
            "mapping_rate": round(n_mapped_ids / n_unique_ids, 4) if n_unique_ids else 0.0,
            "n_edges": n_edges,
            "edge_type_counts": dict(sorted(edge_type_counts.items(), key=lambda item: -item[1])),
            "species": SPECIES,
            "release": RELEASE,
        },
        "parameters": {
            "mapping_file": str(args.mapping_file) if args.mapping_file else None,
            "online_mapping": bool(args.online_mapping),
        },
        "artifacts": {
            "edges": str(output_dir / EDGES_NAME),
            "mapping_cache": str(output_dir / MAPPING_CACHE_NAME),
        },
    }
    manifest_path = output_dir / MANIFEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="OmniPath 下载根目录")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="本地 UniProt->gene 映射缓存（两列 TSV）；缺省时优先读取输出目录已有缓存",
    )
    parser.add_argument(
        "--online-mapping",
        action="store_true",
        help="允许调用 UniProt REST ID-mapping 补齐缺失映射（结果写入映射缓存）",
    )
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output)
    source_path = raw_root / ENZSUB_FILE

    try:
        records = parse_enzsub(source_path)
        uniprot_ids = collect_uniprot_ids(records)

        # Mapping: explicit --mapping-file wins; otherwise reuse the cache
        # written by a previous run; otherwise (opt-in) query online.
        mapping_file = args.mapping_file
        if mapping_file is None and (output_dir / MAPPING_CACHE_NAME).is_file():
            mapping_file = output_dir / MAPPING_CACHE_NAME
        resolved = resolve_genes(
            uniprot_ids,
            mapping_file=mapping_file,
            online=args.online_mapping,
        )

        edges = build_edge_table(records, resolved)
        write_edge_table(output_dir / EDGES_NAME, edges)
        if resolved:
            write_mapping_file(output_dir / MAPPING_CACHE_NAME, resolved)

        edge_type_counts: Dict[str, int] = {}
        for edge in edges:
            edge_type_counts[edge["edge_type"]] = edge_type_counts.get(edge["edge_type"], 0) + 1

        manifest = write_import_manifest(
            output_dir,
            source_path=source_path,
            source_sha256=_sha256_file(source_path),
            n_records=len(records),
            n_unique_ids=len(uniprot_ids),
            n_mapped_ids=len(resolved),
            n_edges=len(edges),
            edge_type_counts=edge_type_counts,
            args=args,
        )
    except ImportError_ as exc:
        print(f"[kinase_substrate] 导入失败：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(manifest),
                "n_records": len(records),
                "n_unique_uniprot_ids": len(uniprot_ids),
                "n_mapped_uniprot_ids": len(resolved),
                "mapping_rate": round(len(resolved) / len(uniprot_ids), 4) if uniprot_ids else 0.0,
                "n_edges": len(edges),
                "edges": str(output_dir / EDGES_NAME),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
