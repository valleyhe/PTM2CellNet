"""ClustalW client for multiple sequence alignment."""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

from .base import (
    BIO_CLUSTAL_APP_AVAILABLE,
    BIO_ALIGN_AVAILABLE,
    ClustalwCommandline,
    ClustalOmegaCommandline,
    SeqIO,
    Seq,
    SeqRecord,
    ToolConfig,
    AlignmentResult,
    _needleman_wunsch,
    _nw_score,
    logger,
)


class ClustalWClient:
    """Client for multiple sequence alignment.

    Tries to use installed ClustalW2 / Clustal Omega first; falls back to a
    progressive pairwise alignment using Needleman-Wunsch.
    """

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        self.config: ToolConfig = config or ToolConfig()

    def check_available(self) -> bool:
        """Return whether ClustalW (or Clustal Omega) is on PATH."""
        if shutil.which("clustalw2") is not None:
            return True
        if shutil.which("clustalo") is not None:
            return True
        if BIO_ALIGN_AVAILABLE:
            return True  # fallback always available
        return False

    def align(self, sequences: Dict[str, str]) -> AlignmentResult:
        """Perform multiple sequence alignment.

        Args:
            sequences: Mapping of sequence names to their amino acid / DNA
                       strings.

        Returns:
            AlignmentResult with keys ``alignment`` (name -> gapped string),
            ``phylogenetic_tree`` (Newick string), and ``consensus``.
        """
        if len(sequences) == 0:
            return AlignmentResult(alignment={}, phylogenetic_tree="", consensus="")

        if len(sequences) == 1:
            only_name = next(iter(sequences))
            only_seq = sequences[only_name]
            return AlignmentResult(
                alignment={only_name: only_seq},
                phylogenetic_tree=f"({only_name});",
                consensus=only_seq,
            )

        # Try external ClustalW first
        result = self._run_external_clustal(sequences)
        if result is not None:
            return result

        logger.info("ClustalW not available; using built-in progressive alignment")
        return self._progressive_align(sequences)

    # ------------------------------------------------------------------
    # External ClustalW
    # ------------------------------------------------------------------

    def _run_external_clustal(self, sequences: Dict[str, str]) -> Optional[AlignmentResult]:
        if not BIO_CLUSTAL_APP_AVAILABLE or not BIO_ALIGN_AVAILABLE:
            return None

        exe = None
        if shutil.which("clustalw2") is not None:
            exe = "clustalw2"
        elif shutil.which("clustalo") is not None:
            exe = "clustalo"

        if exe is None:
            return None

        tmp_dir = tempfile.mkdtemp(prefix="clustal_")
        try:
            fasta_path = os.path.join(tmp_dir, "input.fasta")
            aln_path = os.path.join(tmp_dir, "input.aln")
            tree_path = os.path.join(tmp_dir, "input.dnd")

            records = [
                SeqRecord(Seq(seq), id=name, description="")
                for name, seq in sequences.items()
            ]
            SeqIO.write(records, fasta_path, "fasta")

            if exe == "clustalw2":
                cline = ClustalwCommandline(exe, infile=fasta_path, outfile=aln_path)
            else:
                cline = ClustalOmegaCommandline(
                    exe, infile=fasta_path, outfile=aln_path
                )

            stdout, stderr = cline()

            if os.path.exists(aln_path):
                alignment_dict = self._parse_clustal_aln(aln_path)
            else:
                return None

            tree_newick = ""
            if os.path.exists(tree_path):
                with open(tree_path) as f:
                    tree_newick = f.read().strip()

            consensus = self._compute_consensus(alignment_dict)

            return AlignmentResult(
                alignment=alignment_dict,
                phylogenetic_tree=tree_newick,
                consensus=consensus,
            )
        except (OSError, IOError) as exc:
            logger.warning("External ClustalW file I/O error: %s", exc)
            return None
        except subprocess.SubprocessError as exc:
            logger.warning("External ClustalW subprocess error: %s", exc)
            return None
        except (ValueError, KeyError) as exc:
            logger.warning("External ClustalW parsing error: %s", exc)
            return None
        except (OSError, RuntimeError) as exc:
            logger.error("Unexpected external ClustalW error: %s", exc)
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _parse_clustal_aln(aln_path: str) -> Dict[str, str]:
        """Parse a Clustal-format alignment file into name -> string."""
        result: Dict[str, str] = {}
        current_name = None
        current_seq: List[str] = []
        seq_re = re.compile(r"^\s*(\S+)\s+(\S+)")

        with open(aln_path) as f:
            for line in f:
                if line.startswith("CLUSTAL") or line.strip() == "":
                    continue
                m = seq_re.match(line)
                if m:
                    name, chunk = m.group(1), m.group(2)
                    if name not in result:
                        result[name] = ""
                    result[name] += chunk
                else:
                    continue

        return result

    # ------------------------------------------------------------------
    # Built-in progressive alignment (fallback)
    # ------------------------------------------------------------------

    def _progressive_align(self, sequences: Dict[str, str]) -> AlignmentResult:
        """Progressive multiple sequence alignment using Needleman-Wunsch."""
        names = list(sequences.keys())
        seqs = [sequences[n] for n in names]
        n = len(names)

        # ---- distance matrix (dict-of-dicts for dynamic growth) ----
        dist: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            dist[i] = {}
            for j in range(n):
                if i == j:
                    dist[i][j] = 0.0
                elif j > i:
                    max_score = max(
                        _nw_score(seqs[i], seqs[i]),
                        _nw_score(seqs[j], seqs[j]),
                    )
                    if max_score == 0:
                        d = 1.0
                    else:
                        d = 1.0 - (_nw_score(seqs[i], seqs[j]) / max_score)
                    dist[i][j] = d
                    if j not in dist:
                        dist[j] = {}
                    dist[j][i] = d

        # ---- UPGMA ----
        cluster_ids = list(range(n))
        cluster_sizes: Dict[int, int] = {i: 1 for i in range(n)}
        clusters: Dict[int, Dict[str, str]] = {
            i: {names[i]: seqs[i]} for i in range(n)
        }
        newick_map: Dict[int, str] = {i: names[i] for i in range(n)}
        next_id = n

        while len(cluster_ids) > 1:
            # find closest pair
            min_d = float("inf")
            mi = mj = ii = jj = -1
            for i, a in enumerate(cluster_ids):
                for j, b in enumerate(cluster_ids):
                    if i >= j:
                        continue
                    if dist[a][b] < min_d:
                        min_d = dist[a][b]
                        mi, mj = a, b
                        ii, jj = i, j

            # merge clusters[mi] and clusters[mj]
            merged = self._merge_two_clusters(clusters[mi], clusters[mj])
            new_id = next_id
            next_id += 1
            clusters[new_id] = merged

            brlen = f"{min_d / 2:.4f}"
            newick_map[new_id] = (
                f"({newick_map[mi]}:{brlen},{newick_map[mj]}:{brlen})"
            )

            # update distances
            dist[new_id] = {}
            for k in cluster_ids:
                if k in (mi, mj):
                    continue
                d = (
                    dist[mi][k] * cluster_sizes[mi]
                    + dist[mj][k] * cluster_sizes[mj]
                ) / (cluster_sizes[mi] + cluster_sizes[mj])
                dist[new_id][k] = d
                dist[k][new_id] = d

            cluster_sizes[new_id] = cluster_sizes[mi] + cluster_sizes[mj]

            cluster_ids.pop(max(ii, jj))
            cluster_ids.pop(min(ii, jj))
            cluster_ids.append(new_id)

        root = cluster_ids[0]
        tree_newick = newick_map[root] + ";"

        alignment_dict = clusters[root]
        consensus = self._compute_consensus(alignment_dict)

        return AlignmentResult(
            alignment=alignment_dict,
            phylogenetic_tree=tree_newick,
            consensus=consensus,
        )

    @staticmethod
    def _merge_two_clusters(
        aln_a: Dict[str, str], aln_b: Dict[str, str]
    ) -> Dict[str, str]:
        """Align two clusters by aligning their representatives."""
        rep_a_name = next(iter(aln_a))
        rep_b_name = next(iter(aln_b))
        raw_a = aln_a[rep_a_name].replace("-", "")
        raw_b = aln_b[rep_b_name].replace("-", "")

        gapped_a, gapped_b = _needleman_wunsch(raw_a, raw_b)
        length = len(gapped_a)

        merged: Dict[str, str] = {}

        for name, seq in aln_a.items():
            raw = seq.replace("-", "")
            merged[name] = ClustalWClient._apply_gap_pattern(
                raw, raw_a, gapped_a, length
            )

        for name, seq in aln_b.items():
            raw = seq.replace("-", "")
            merged[name] = ClustalWClient._apply_gap_pattern(
                raw, raw_b, gapped_b, length
            )

        return merged

    @staticmethod
    def _apply_gap_pattern(
        raw: str, raw_ref: str, gapped_ref: str, target_len: int
    ) -> str:
        """Apply the gap pattern from *gapped_ref* to *raw*.

        *raw_ref* is the ungapped version of *gapped_ref*.
        """
        result_chars = []
        idx = 0
        for ch in gapped_ref:
            if ch == "-":
                result_chars.append("-")
            else:
                result_chars.append(raw[idx])
                idx += 1
        while len(result_chars) < target_len:
            result_chars.append("-")
        return "".join(result_chars)

    @staticmethod
    def _compute_consensus(alignment: Dict[str, str]) -> str:
        """Compute a simple majority-rule consensus from an alignment."""
        if not alignment:
            return ""

        seqs = list(alignment.values())
        length = len(seqs[0])
        consensus_chars = []

        for pos in range(length):
            counts: Dict[str, int] = {}
            for seq in seqs:
                if pos < len(seq):
                    ch = seq[pos]
                    counts[ch] = counts.get(ch, 0) + 1
            if not counts:
                consensus_chars.append("-")
            else:
                best = max(counts, key=lambda k: counts[k])
                consensus_chars.append(best)

        return "".join(consensus_chars)
