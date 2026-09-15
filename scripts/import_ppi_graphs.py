#!/usr/bin/env python3
"""Import STRING / BioPlex / RegNetwork raw snapshots into canonical edge tables.

F-03（余量）：消费 ``data/raw/{string,bioplex,regnetwork}`` 三类在盘图数据，
产出下游（``src/analysis/pathway_integration.py``、跨尺度 NPZ 组装）可消费的
canonical signaling-edge 产物：

* ``<source>_edges.tsv`` — 统一 7 列边表
  ``source_gene, target_gene, edge_type, score, evidence, species, release``；
* ``import_manifest.json`` — provenance（schema
  ``ptm2cellnet.ppi-graph.import.v1``：源 sha256、解析统计、词表版本、
  参数与产物清单）。

各数据源说明
------------
* ``bioplex`` — ``bioplex_293T_v2_edges.tsv``（两列基因符号，117,930 边）。
  无需 ID 映射，直接转 canonical。
* ``regnetwork`` — ``human.source`` 四列
  ``TF_symbol <TAB> TF_Entrez <TAB> Target_symbol <TAB> Target_Entrez``
  （CRLF 行尾）；符号已在文件内。``human.core.7z`` 解压得到
  ``human.core.txt``（六列，2026-08-17 在线核对）：
  ``regulator_symbol <TAB> regulator_id <TAB> target_symbol <TAB>
  target_entrez <TAB> regulator_type <TAB> target_type``，regulator_type ∈
  {TF, miRNA, lncRNA, circRNA}。RegNetwork 批量发布**不含
  activation/inhibition 符号**——方向语义不存在于数据中，不捏造；
  core 的价值是按 ``(regulator_id, target_entrez)`` join 提供调控者类型，
  用于输出 ``{tf|mirna|lncrna|circrna}_regulation:unspecified`` 边型。
  未 join 命中的行按命名规约兜底（``hsa-miR-*`` → mirna），
  其余记 ``regulation:unspecified``；manifest 显式记录——不静默、不丢弃边。
* ``string`` — ``9606.protein.links.v12.0.txt.gz``（默认 physical 子网，
  ``--string-full`` 切全量）按 ``--min-score``（默认 700）过滤
  ``combined_score``。节点为 ``9606.ENSP...`` 蛋白 ID，需要
  ``--ensp-mapping``（两列 TSV：``STRING_protein_id <TAB> gene_symbol``，
  可从 STRING ``protein.info`` 文件构造）；未映射 ID 原样保留（与
  kinase_substrate 导入器同策略），manifest 记录 mapping_rate。

Design notes
------------
* Pure stdlib；gzip 按需导入。
* Fail-fast：缺输入/列头不符/映射文件格式错误均报错退出（带 path:line）。
* 输出列统一，三个源的 edge_type 词汇表记录在 manifest
  （``edge_type_vocab`` / ``relation_vocab``）。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SOURCES = ("bioplex", "regnetwork", "string")

MANIFEST_SCHEMA = "ptm2cellnet.ppi-graph.import.v1"
EDGE_TABLE_COLUMNS = (
    "source_gene",
    "target_gene",
    "edge_type",
    "score",
    "evidence",
    "species",
    "release",
)

DEFAULT_RAW_ROOTS = {
    "bioplex": "data/raw/bioplex",
    "regnetwork": "data/raw/regnetwork",
    "string": "data/raw/string",
}
DEFAULT_OUTPUT = "data/processed"

SPECIES = "9606"

#: edge_type 词汇表（manifest 记录，供 edge_type_vocab_versioned 检查）。
EDGE_TYPE_VOCAB = {
    "bioplex": ("ppi:bioplex_apms",),
    "regnetwork": (
        "tf_regulation:unspecified",
        "mirna_regulation:unspecified",
        "lncrna_regulation:unspecified",
        "circrna_regulation:unspecified",
        "regulation:unspecified",
    ),
    "string": ("ppi:string_physical", "ppi:string_combined"),
}

#: RegNetwork human.core.txt 的 regulator_type → edge_type 前缀。
#: RegNetwork 批量文件不发布 activation/inhibition 符号，方向一律 unspecified。
REGNETWORK_REGULATOR_TO_EDGE_PREFIX = {
    "TF": "tf_regulation",
    "miRNA": "mirna_regulation",
    "lncRNA": "lncrna_regulation",
    "circRNA": "circrna_regulation",
}


class ImportError_(RuntimeError):
    """Raised when a required raw input is missing or malformed."""


# ---------------------------------------------------------------------------
# sha256 / 通用解析
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# BioPlex
# ---------------------------------------------------------------------------


def parse_bioplex(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    """解析 BioPlex 293T v2 边表（header: source<TAB>target，基因符号）。"""
    if not path.is_file():
        raise ImportError_(f"缺少 BioPlex 输入文件: {path}")
    edges: List[Dict[str, str]] = []
    with open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            cells = [cell.strip() for cell in line.split("\t")]
            if line_no == 1:
                if cells[:2] != ["source", "target"]:
                    raise ImportError_(f"BioPlex 列头不符合契约（期望 source/target）: {path}:1 -> {cells[:2]!r}")
                continue
            if len(cells) < 2 or not cells[0] or not cells[1]:
                raise ImportError_(f"BioPlex 行列数不足或符号为空: {path}:{line_no}")
            edges.append(
                {
                    "source_gene": cells[0],
                    "target_gene": cells[1],
                    "edge_type": "ppi:bioplex_apms",
                    "score": "",
                    "evidence": "BioPlex 2.0 HEK293T AP-MS (Huttlin et al. 2017, NDEx release)",
                    "species": SPECIES,
                    "release": "bioplex3-293T-v2",
                }
            )
            if max_rows is not None and len(edges) >= max_rows:
                break
    if not edges:
        raise ImportError_(f"BioPlex 文件无数据行: {path}")
    return edges


# ---------------------------------------------------------------------------
# RegNetwork
# ---------------------------------------------------------------------------


def load_entrez_symbol_map(node_path: Path) -> Dict[str, str]:
    """解析 human.node（Entrez <TAB> symbol）为回填映射。"""
    mapping: Dict[str, str] = {}
    if not node_path.is_file():
        raise ImportError_(f"缺少 RegNetwork human.node: {node_path}")
    with open(node_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            cells = [cell.strip() for cell in line.split("\t")]
            if len(cells) < 2 or not cells[0] or not cells[1]:
                raise ImportError_(f"human.node 行格式错误: {node_path}:{line_no}")
            mapping[cells[0]] = cells[1]
    if not mapping:
        raise ImportError_(f"human.node 无有效行: {node_path}")
    return mapping


def load_regnetwork_core(core_path: Optional[Path]) -> Tuple[Dict[Tuple[str, str], str], int]:
    """解析已解压的 human.core.txt（六列真实格式，2026-08-17 在线核对）。

    每行 ``regulator_symbol <TAB> regulator_id <TAB> target_symbol <TAB>
    target_entrez <TAB> regulator_type <TAB> target_type``；regulator_type ∈
    {TF, miRNA, lncRNA, circRNA}。返回 ``(regulator_id, target_entrez) ->
    regulator_type`` 映射与 join 键缺失被跳过的行数（811,766 行实测 1 行
    circRNA 目标无 Entrez——跳过并显式计数，不静默）。RegNetwork 批量
    文件不发布 activation/inhibition 符号——方向语义不存在于数据中，
    不捏造。路径不可用返回空映射（调用方按命名规约兜底，manifest 记录）。
    """
    relations: Dict[Tuple[str, str], str] = {}
    n_keyless = 0
    if core_path is None or not Path(core_path).is_file():
        return relations, n_keyless
    with open(Path(core_path), "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            cells = [cell.strip() for cell in line.split("\t")]
            if len(cells) < 6 or not cells[4]:
                raise ImportError_(f"human.core.txt 行格式错误（期望 6 列）: {core_path}:{line_no}")
            regulator_type = cells[4]
            if regulator_type not in REGNETWORK_REGULATOR_TO_EDGE_PREFIX:
                raise ImportError_(f"human.core.txt 未知 regulator_type {regulator_type!r}: {core_path}:{line_no}")
            if not cells[1] or not cells[3]:
                n_keyless += 1
                continue
            relations[(cells[1], cells[3])] = regulator_type
    if not relations:
        raise ImportError_(f"human.core.txt 无有效行: {core_path}")
    return relations, n_keyless


def _regnetwork_edge_type(regulator_symbol: str, regulator_type: Optional[str]) -> str:
    """按 core 类型（或命名规约）决定 RegNetwork 边的 edge_type。

    core 命中 → ``{tf|mirna|lncrna|circrna}_regulation:unspecified``；
    未命中 → ``hsa-miR-*`` 前缀是 miRNA 命名标准，按规约归类；其余无
    类型证据，记 neutral 的 ``regulation:unspecified``（不冒充 TF）。
    """
    if regulator_type is not None:
        prefix = REGNETWORK_REGULATOR_TO_EDGE_PREFIX[regulator_type]
        return f"{prefix}:unspecified"
    if regulator_symbol.startswith("hsa-miR"):
        return "mirna_regulation:unspecified"
    return "regulation:unspecified"


def parse_regnetwork(
    source_path: Path,
    node_path: Path,
    core_path: Optional[Path] = None,
    max_rows: Optional[int] = None,
) -> tuple:
    """解析 human.source 四列调控边，返回 (edges, typing_source_note, n_typed_by_core)。"""
    if not source_path.is_file():
        raise ImportError_(f"缺少 RegNetwork human.source: {source_path}")
    fallback = load_entrez_symbol_map(node_path)
    regulator_types, n_core_keyless = load_regnetwork_core(core_path)
    typing_note = (
        f"human.core.txt: {core_path} (regulator typing; RegNetwork publishes no "
        "activation/inhibition signs in bulk files; "
        f"{n_core_keyless} core rows skipped for missing join key)"
        if regulator_types
        else "unavailable (human.core.txt not provided; naming-convention fallback)"
    )
    edges: List[Dict[str, str]] = []
    n_typed_by_core = 0
    with open(source_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            cells = [cell.strip() for cell in line.split("\t")]
            if len(cells) < 4:
                raise ImportError_(f"human.source 行列数不足（期望 4，实际 {len(cells)}）: {source_path}:{line_no}")
            tf_symbol, tf_entrez, target_symbol, target_entrez = cells[:4]
            # 符号缺失（空或 '-'）时用 human.node 的 Entrez->symbol 回填
            if not tf_symbol or tf_symbol == "-":
                tf_symbol = fallback.get(tf_entrez, tf_entrez)
            if not target_symbol or target_symbol == "-":
                target_symbol = fallback.get(target_entrez, target_entrez)
            regulator_type = regulator_types.get((tf_entrez, target_entrez))
            if regulator_type is not None:
                n_typed_by_core += 1
            edge_type = _regnetwork_edge_type(tf_symbol, regulator_type)
            edges.append(
                {
                    "source_gene": tf_symbol,
                    "target_gene": target_symbol,
                    "edge_type": edge_type,
                    "score": "",
                    "evidence": "RegNetwork-1.0 human (Liu et al. 2015 Database)",
                    "species": SPECIES,
                    "release": "RegNetwork-1.0",
                }
            )
            if max_rows is not None and len(edges) >= max_rows:
                break
    if not edges:
        raise ImportError_(f"human.source 无数据行: {source_path}")
    return edges, typing_note, n_typed_by_core


# ---------------------------------------------------------------------------
# STRING
# ---------------------------------------------------------------------------


def load_ensp_mapping(path: Optional[Path]) -> Dict[str, str]:
    """加载 STRING protein ID -> gene symbol 映射（两列 TSV）。

    键同时登记全 ID（如 ``9606.ENSP00000000233``）与裸 ENSP
    （``ENSP00000000233``），兼容 STRING protein.info 与自建映射两种风格。
    """
    mapping: Dict[str, str] = {}
    if path is None:
        return mapping
    p = Path(path)
    if not p.is_file():
        raise ImportError_(f"ENSP 映射文件不存在: {p}")
    with open(p, "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.startswith("#"):
                continue
            cells = [cell.strip() for cell in line.split("\t")]
            if len(cells) < 2 or not cells[0] or not cells[1]:
                raise ImportError_(f"ENSP 映射行格式错误（期望 string_protein_id<TAB>gene_symbol）: {p}:{line_no}")
            protein_id, symbol = cells[0], cells[1]
            mapping[protein_id] = symbol
            if "." in protein_id:
                mapping[protein_id.split(".", 1)[1]] = symbol
    if not mapping:
        raise ImportError_(f"ENSP 映射文件无有效行: {p}")
    return mapping


def parse_string(
    path: Path,
    mapping: Mapping[str, str],
    *,
    min_score: int = 700,
    max_rows: Optional[int] = None,
) -> tuple:
    """解析 STRING links（gz TSV：protein1 protein2 combined_score）。

    返回 (edges, n_below_threshold, n_unmapped)；未映射 ID 原样保留。
    """
    if not path.is_file():
        raise ImportError_(f"缺少 STRING 输入文件: {path}")
    edge_type = "ppi:string_physical" if "physical" in path.name else "ppi:string_combined"
    edges: List[Dict[str, str]] = []
    n_below = 0
    unmapped: set = set()
    with _open_maybe_gzip(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            cells = line.split()
            if line_no == 1:
                if cells[:3] != ["protein1", "protein2", "combined_score"]:
                    raise ImportError_(f"STRING 列头不符合契约: {path}:1 -> {cells[:3]!r}")
                continue
            if len(cells) < 3:
                raise ImportError_(f"STRING 行列数不足: {path}:{line_no}")
            try:
                score = int(cells[2])
            except ValueError as exc:
                raise ImportError_(f"STRING combined_score 非整数: {path}:{line_no}") from exc
            if score < min_score:
                n_below += 1
                continue
            protein1, protein2 = cells[0], cells[1]
            symbol1 = mapping.get(protein1)
            symbol2 = mapping.get(protein2)
            if symbol1 is None:
                unmapped.add(protein1)
            if symbol2 is None:
                unmapped.add(protein2)
            edges.append(
                {
                    "source_gene": symbol1 or protein1,
                    "target_gene": symbol2 or protein2,
                    "edge_type": edge_type,
                    "score": str(score),
                    "evidence": "STRING v12.0 human combined_score (experimental/database/textmining composite)",
                    "species": SPECIES,
                    "release": "string-v12.0",
                }
            )
            if max_rows is not None and len(edges) >= max_rows:
                break
    if not edges:
        raise ImportError_(f"STRING 过滤后无数据行（min_score={min_score}）: {path}")
    return edges, n_below, unmapped


# ---------------------------------------------------------------------------
# Canonical 输出
# ---------------------------------------------------------------------------


def write_edge_table(path: Path, edges: Sequence[Mapping[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(EDGE_TABLE_COLUMNS) + "\n")
        for edge in edges:
            handle.write("\t".join(str(edge[column]) for column in EDGE_TABLE_COLUMNS) + "\n")


def write_import_manifest(
    output_dir: Path,
    *,
    source_name: str,
    source_files: Sequence[Path],
    summary: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source": {
            "name": source_name,
            "files": [{"file": str(path), "sha256": _sha256_file(path)} for path in source_files],
        },
        "summary": dict(summary),
        "parameters": dict(parameters),
        "edge_type_vocab": list(EDGE_TYPE_VOCAB[source_name]),
        "relation_vocab": (
            {k: f"{v}:unspecified" for k, v in REGNETWORK_REGULATOR_TO_EDGE_PREFIX.items()}
            if source_name == "regnetwork"
            else None
        ),
        "artifacts": {
            "edges": str(output_dir / f"{source_name}_edges.tsv"),
        },
    }
    manifest_path = output_dir / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Import STRING/BioPlex/RegNetwork into canonical edge tables (F-03)")
    parser.add_argument("--source", required=True, choices=SOURCES, help="数据源")
    parser.add_argument("--raw-root", default=None, help="覆盖默认 raw 目录")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT, help="processed 输出根目录")
    parser.add_argument("--max-rows", type=int, default=None, help="调试用：限制解析行数")
    # STRING 参数
    parser.add_argument(
        "--string-variant",
        choices=("physical", "full"),
        default="physical",
        help="STRING 输入：physical 子网（默认）或全量 links",
    )
    parser.add_argument("--min-score", type=int, default=700, help="STRING combined_score 阈值（默认 700）")
    parser.add_argument(
        "--ensp-mapping",
        type=Path,
        default=None,
        help="STRING protein.info 风格映射 TSV（string_protein_id<TAB>gene_symbol）",
    )
    # RegNetwork 参数
    parser.add_argument(
        "--regnetwork-core",
        type=Path,
        default=None,
        help="已解压的 human.core.txt（6 列，提供 regulator 类型标注）；缺省时按命名规约兜底",
    )
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root or DEFAULT_RAW_ROOTS[args.source])
    output_dir = Path(args.output_root) / args.source

    try:
        if args.source == "bioplex":
            source_files = [raw_root / "bioplex_293T_v2_edges.tsv"]
            edges = parse_bioplex(source_files[0], max_rows=args.max_rows)
            summary = {"n_records": len(edges), "n_edges": len(edges)}
            parameters: Dict[str, Any] = {}

        elif args.source == "regnetwork":
            source_files = [raw_root / "human.source", raw_root / "human.node"]
            edges, typing_note, n_typed = parse_regnetwork(
                source_files[0], source_files[1], args.regnetwork_core, max_rows=args.max_rows
            )
            summary = {
                "n_records": len(edges),
                "n_edges": len(edges),
                "n_typed_by_core": n_typed,
                "typing_source": typing_note,
            }
            parameters = {
                "regnetwork_core": str(args.regnetwork_core) if args.regnetwork_core else None,
            }

        else:  # string
            filename = (
                "9606.protein.physical.links.v12.0.txt.gz"
                if args.string_variant == "physical"
                else "9606.protein.links.v12.0.txt.gz"
            )
            source_files = [raw_root / filename]
            mapping = load_ensp_mapping(args.ensp_mapping)
            edges, n_below, unmapped = parse_string(
                source_files[0], mapping, min_score=args.min_score, max_rows=args.max_rows
            )
            n_nodes = len({e["source_gene"] for e in edges} | {e["target_gene"] for e in edges})
            summary = {
                "n_edges": len(edges),
                "n_below_threshold": n_below,
                "n_unique_nodes": n_nodes,
                "n_unmapped_nodes": len(unmapped),
                "mapping_rate": round(1 - len(unmapped) / n_nodes, 4) if n_nodes else 0.0,
            }
            parameters = {
                "string_variant": args.string_variant,
                "min_score": args.min_score,
                "ensp_mapping": str(args.ensp_mapping) if args.ensp_mapping else None,
            }

        write_edge_table(output_dir / f"{args.source}_edges.tsv", edges)
        edge_type_counts: Dict[str, int] = {}
        for edge in edges:
            edge_type_counts[edge["edge_type"]] = edge_type_counts.get(edge["edge_type"], 0) + 1
        summary["edge_type_counts"] = dict(sorted(edge_type_counts.items(), key=lambda item: -item[1]))
        manifest = write_import_manifest(
            output_dir,
            source_name=args.source,
            source_files=source_files,
            summary=summary,
            parameters=parameters,
        )
    except ImportError_ as exc:
        print(f"[{args.source}] 导入失败：{exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {"ok": True, "source": args.source, "manifest": str(manifest), **summary},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
