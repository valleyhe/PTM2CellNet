"""External bioinformatics tool integration.

Provides interface classes for external bioinformatics tools (AlphaFold, BLAST,
ClustalW, PSIPRED) with real implementations that use public APIs where
available and fall back to built-in methods when external tools are not installed.
"""

import logging
import re
import tempfile
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency checks (module-level)
# ---------------------------------------------------------------------------

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from Bio.Blast import NCBIWWW, NCBIXML
    BIO_BLAST_AVAILABLE = True
except ImportError:
    BIO_BLAST_AVAILABLE = False

try:
    from Bio.Align.Applications import ClustalwCommandline, ClustalOmegaCommandline
    BIO_CLUSTAL_APP_AVAILABLE = True
except ImportError:
    BIO_CLUSTAL_APP_AVAILABLE = False

try:
    from Bio import Align
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIO_ALIGN_AVAILABLE = True
except ImportError:
    BIO_ALIGN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA_THREE_LETTER: Dict[str, str] = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}

# Chou-Fasman propensity tables (normalized)
_CHOU_FASMAN_HELIX: Dict[str, float] = {
    'A': 1.42, 'C': 0.77, 'D': 1.01, 'E': 1.37, 'F': 1.13,
    'G': 0.57, 'H': 1.08, 'I': 1.09, 'K': 1.23, 'L': 1.30,
    'M': 1.20, 'N': 0.79, 'P': 0.52, 'Q': 1.17, 'R': 1.15,
    'S': 0.79, 'T': 0.82, 'V': 1.02, 'W': 1.14, 'Y': 0.74,
}

_CHOU_FASMAN_SHEET: Dict[str, float] = {
    'A': 0.83, 'C': 1.30, 'D': 0.80, 'E': 1.02, 'F': 1.39,
    'G': 0.57, 'H': 0.87, 'I': 1.40, 'K': 0.81, 'L': 1.22,
    'M': 1.30, 'N': 0.72, 'P': 0.64, 'Q': 1.12, 'R': 0.99,
    'S': 0.79, 'T': 1.05, 'V': 1.43, 'W': 1.27, 'Y': 1.32,
}

_CHOU_FASMAN_COIL: Dict[str, float] = {
    'A': 0.75, 'C': 0.93, 'D': 1.19, 'E': 0.61, 'F': 0.48,
    'G': 1.86, 'H': 1.05, 'I': 0.47, 'K': 1.02, 'L': 0.48,
    'M': 0.60, 'N': 1.49, 'P': 1.84, 'Q': 0.71, 'R': 0.85,
    'S': 1.40, 'T': 1.13, 'V': 0.41, 'W': 0.59, 'Y': 0.94,
}


# ---------------------------------------------------------------------------
# Helper: Needleman-Wunsch global alignment  (fallback when ClustalW absent)
# ---------------------------------------------------------------------------

def _needleman_wunsch(seq1: str, seq2: str,
                      match: int = 2, mismatch: int = -1, gap: int = -2
                      ) -> Tuple[str, str]:
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )
    i, j = n, m
    a1_chars, a2_chars = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                match if seq1[i - 1] == seq2[j - 1] else mismatch):
            a1_chars.append(seq1[i - 1])
            a2_chars.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            a1_chars.append(seq1[i - 1])
            a2_chars.append('-')
            i -= 1
        else:
            a1_chars.append('-')
            a2_chars.append(seq2[j - 1])
            j -= 1
    return ''.join(reversed(a1_chars)), ''.join(reversed(a2_chars))


def _nw_score(seq1: str, seq2: str,
              match: int = 2, mismatch: int = -1, gap: int = -2) -> int:
    """Return the Needleman-Wunsch optimal score (DP table value)."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(2)]
    for j in range(m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        dp[1][0] = i * gap
        for j in range(1, m + 1):
            score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[1][j] = max(
                dp[0][j - 1] + score,
                dp[0][j] + gap,
                dp[1][j - 1] + gap,
            )
        dp[0], dp[1] = dp[1], dp[0]
    return dp[0][m]


# ---------------------------------------------------------------------------
# 1. AlphaFoldClient
# ---------------------------------------------------------------------------

class AlphaFoldClient:
    """Client for AlphaFold protein structure prediction.

    Uses the public EBI AlphaFold Database API to fetch pre-computed structures.
    Falls back to a simple extended-chain PDB when the API is unreachable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    def check_available(self) -> bool:
        """Return whether the AlphaFold API is reachable."""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            resp = requests.get("https://alphafold.ebi.ac.uk/api", timeout=5)
            return resp.ok
        except Exception:
            return False

    def predict_structure(
        self, sequence: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Predict the 3D structure of a protein.

        Tries the EBI AlphaFold Database API first (requires *uniprot_id* in
        *kwargs*).  Falls back to a simple extended-chain PDB string.

        Args:
            sequence: Amino acid sequence (single-letter codes).
            **kwargs: May include ``uniprot_id`` (str) for the EBI API lookup.

        Returns:
            Dict with keys ``pdb_string``, ``confidence``, and
            ``predicted_aligned_error``.
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests module not available; using fallback PDB")
            return self._fallback_pdb(sequence)

        uniprot_id = kwargs.get("uniprot_id", "")
        if uniprot_id:
            try:
                url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
                resp = requests.get(url, timeout=30)
                if resp.ok:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        entry = data[0]
                    else:
                        entry = data

                    pdb_url = entry.get("pdbUrl", "")
                    confidence = float(entry.get("confidence", 0.0))
                    pae = entry.get("predictedAlignedError", [])

                    if pdb_url:
                        pdb_resp = requests.get(pdb_url, timeout=60)
                        if pdb_resp.ok:
                            return {
                                "pdb_string": pdb_resp.text,
                                "confidence": confidence / 100.0 if confidence > 1.0 else confidence,
                                "predicted_aligned_error": pae,
                            }
            except Exception as exc:
                logger.warning("AlphaFold EBI API error: %s", exc)

        logger.info("Using fallback PDB generation")
        return self._fallback_pdb(sequence)

    def _fallback_pdb(self, sequence: str) -> Dict[str, Any]:
        """Generate a simple extended-chain PDB string."""
        lines = []
        x, y, z = 0.0, 0.0, 0.0
        n = len(sequence)
        plddt = 50.0
        confidence = 0.5

        for i, aa in enumerate(sequence):
            res_name = _AA_THREE_LETTER.get(aa.upper(), 'UNK')
            x += 3.8
            y = 0.5 * ((i % 4) - 1.5)
            z = 0.3 * ((i % 3) - 1)
            occ = 1.00
            b = plddt * (1.0 - 0.5 * (i / max(n, 1)))
            serial = i + 1
            resnum = i + 1
            lines.append(
                f"ATOM  {serial:5d}  CA  {res_name:3s} A{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          CA"
            )
        pdb_str = "\n".join(lines)

        return {
            "pdb_string": pdb_str,
            "confidence": confidence,
            "predicted_aligned_error": [],
        }


# ---------------------------------------------------------------------------
# 2. BLASTClient
# ---------------------------------------------------------------------------

class BLASTClient:
    """Client for NCBI BLAST sequence similarity search.

    Uses BioPython's ``NCBIWWW.qblast`` to query NCBI remotely.  Falls back
    to an empty result set on failure.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    def check_available(self) -> bool:
        """Return whether BioPython BLAST modules are available."""
        return BIO_BLAST_AVAILABLE

    def search(
        self,
        sequence: str,
        database: str = "nr",
        e_value: float = 0.001,
    ) -> List[Dict[str, Any]]:
        """Search for similar sequences using BLAST.

        Args:
            sequence: Query amino acid or nucleotide sequence.
            database: BLAST database name (default ``"nr"``).
            e_value: Expect-value threshold for reporting hits.

        Returns:
            List of dicts with keys ``accession``, ``description``, ``e_value``,
            ``score``, ``identity``.  Returns empty list on failure.
        """
        if not BIO_BLAST_AVAILABLE:
            logger.warning("BioPython BLAST not available; returning []")
            return []

        try:
            result_handle = NCBIWWW.qblast(
                "blastp", database, sequence, expect=e_value
            )
            blast_records = NCBIXML.parse(result_handle)

            hits: List[Dict[str, Any]] = []
            for record in blast_records:
                for alignment in record.alignments[:20]:
                    for hsp in alignment.hsps:
                        hits.append({
                            "accession": alignment.accession,
                            "description": alignment.title,
                            "e_value": hsp.expect,
                            "score": hsp.score,
                            "identity": f"{hsp.identities}/{hsp.align_length}",
                        })
                    if len(hits) >= 50:
                        break
                if len(hits) >= 50:
                    break

            result_handle.close()
            return hits

        except Exception as exc:
            logger.warning("BLAST search error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# 3. ClustalWClient
# ---------------------------------------------------------------------------

class ClustalWClient:
    """Client for multiple sequence alignment.

    Tries to use installed ClustalW2 / Clustal Omega first; falls back to a
    progressive pairwise alignment using Needleman-Wunsch.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    def check_available(self) -> bool:
        """Return whether ClustalW (or Clustal Omega) is on PATH."""
        if shutil.which("clustalw2") is not None:
            return True
        if shutil.which("clustalo") is not None:
            return True
        if BIO_ALIGN_AVAILABLE:
            return True  # fallback always available
        return False

    def align(
        self, sequences: Dict[str, str]
    ) -> Dict[str, Any]:
        """Perform multiple sequence alignment.

        Args:
            sequences: Mapping of sequence names to their amino acid / DNA
                       strings.

        Returns:
            Dict with keys ``alignment`` (name -> gapped string),
            ``phylogenetic_tree`` (Newick string), and ``consensus``.
        """
        if len(sequences) == 0:
            return {"alignment": {}, "phylogenetic_tree": "", "consensus": ""}

        if len(sequences) == 1:
            only_name = next(iter(sequences))
            only_seq = sequences[only_name]
            return {
                "alignment": {only_name: only_seq},
                "phylogenetic_tree": f"({only_name});",
                "consensus": only_seq,
            }

        # Try external ClustalW first
        result = self._run_external_clustal(sequences)
        if result is not None:
            return result

        logger.info("ClustalW not available; using built-in progressive alignment")
        return self._progressive_align(sequences)

    # ------------------------------------------------------------------
    # External ClustalW
    # ------------------------------------------------------------------

    def _run_external_clustal(self, sequences: Dict[str, str]) -> Optional[Dict[str, Any]]:
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

            return {
                "alignment": alignment_dict,
                "phylogenetic_tree": tree_newick,
                "consensus": consensus,
            }
        except Exception as exc:
            logger.warning("External ClustalW error: %s", exc)
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _parse_clustal_aln(aln_path: str) -> Dict[str, str]:
        """Parse a Clustal-format alignment file into name -> string."""
        result: Dict[str, str] = {}
        current_name = None
        current_seq = []
        seq_re = re.compile(r'^\s*(\S+)\s+(\S+)')

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

    def _progressive_align(self, sequences: Dict[str, str]) -> Dict[str, Any]:
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
            min_d = float('inf')
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

        return {
            "alignment": alignment_dict,
            "phylogenetic_tree": tree_newick,
            "consensus": consensus,
        }

    @staticmethod
    def _merge_two_clusters(
        aln_a: Dict[str, str], aln_b: Dict[str, str]
    ) -> Dict[str, str]:
        """Align two clusters by aligning their representatives."""
        rep_a_name = next(iter(aln_a))
        rep_b_name = next(iter(aln_b))
        raw_a = aln_a[rep_a_name].replace('-', '')
        raw_b = aln_b[rep_b_name].replace('-', '')

        gapped_a, gapped_b = _needleman_wunsch(raw_a, raw_b)
        length = len(gapped_a)

        merged: Dict[str, str] = {}

        for name, seq in aln_a.items():
            raw = seq.replace('-', '')
            merged[name] = ClustalWClient._apply_gap_pattern(raw, raw_a, gapped_a, length)

        for name, seq in aln_b.items():
            raw = seq.replace('-', '')
            merged[name] = ClustalWClient._apply_gap_pattern(raw, raw_b, gapped_b, length)

        return merged

    @staticmethod
    def _apply_gap_pattern(raw: str, raw_ref: str, gapped_ref: str,
                           target_len: int) -> str:
        """Apply the gap pattern from *gapped_ref* to *raw*.

        *raw_ref* is the ungapped version of *gapped_ref*.
        """
        result_chars = []
        idx = 0
        for ch in gapped_ref:
            if ch == '-':
                result_chars.append('-')
            else:
                result_chars.append(raw[idx])
                idx += 1
        while len(result_chars) < target_len:
            result_chars.append('-')
        return ''.join(result_chars)

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
                consensus_chars.append('-')
            else:
                best = max(counts, key=counts.get)
                consensus_chars.append(best)

        return ''.join(consensus_chars)


# ---------------------------------------------------------------------------
# 4. PSIPREDClient
# ---------------------------------------------------------------------------

class PSIPREDClient:
    """Client for protein secondary structure prediction.

    Implements the Chou-Fasman statistical method with a sliding-window
    smoother as a built-in predictor.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    def check_available(self) -> bool:
        """Always available (built-in predictor)."""
        return True

    def predict_secondary_structure(
        self, sequence: str
    ) -> Dict[str, Any]:
        """Predict secondary structure using Chou-Fasman propensities.

        Uses a sliding window of length 7 to smooth per-residue predictions.

        Args:
            sequence: Amino acid sequence (single-letter codes).

        Returns:
            Dict with keys:
                - ``ss_prediction``: string of ``H`` (helix), ``E`` (sheet),
                  ``C`` (coil), one per residue.
                - ``confidence_scores``: list of floats in [0, 1].
        """
        seq = sequence.upper()
        n = len(seq)
        if n == 0:
            return {"ss_prediction": "", "confidence_scores": []}

        window = 7
        half = window // 2

        ss_chars: List[str] = []
        confs: List[float] = []

        for i in range(n):
            # collect window
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window_chars = seq[start:end]

            h_prop = sum(_CHOU_FASMAN_HELIX.get(ch, 0.0) for ch in window_chars)
            e_prop = sum(_CHOU_FASMAN_SHEET.get(ch, 0.0) for ch in window_chars)
            c_prop = sum(_CHOU_FASMAN_COIL.get(ch, 0.0) for ch in window_chars)

            props = [h_prop, e_prop, c_prop]
            labels = ['H', 'E', 'C']

            sorted_props = sorted(props, reverse=True)
            best = max(range(3), key=lambda k: props[k])
            second_best = sorted_props[1] if len(sorted_props) > 1 else 0.0

            confidence = min(1.0, max(0.0, sorted_props[0] - second_best) / max(window, 1))

            ss_chars.append(labels[best])
            confs.append(round(confidence, 4))

        ss_prediction = ''.join(ss_chars)

        return {
            "ss_prediction": ss_prediction,
            "confidence_scores": confs,
        }
