"""PSIPRED client for secondary structure prediction (Chou-Fasman)."""

from typing import List, Optional

from .base import (
    ToolConfig,
    SecondaryStructurePrediction,
    _CHOU_FASMAN_HELIX,
    _CHOU_FASMAN_SHEET,
    _CHOU_FASMAN_COIL,
    logger,
)


class PSIPREDClient:
    """Client for protein secondary structure prediction.

    Implements the Chou-Fasman statistical method with a sliding-window
    smoother as a built-in predictor.
    """

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        self.config: ToolConfig = config or ToolConfig()

    def check_available(self) -> bool:
        """Always available (built-in predictor)."""
        return True

    def predict_secondary_structure(
        self, sequence: str
    ) -> SecondaryStructurePrediction:
        """Predict secondary structure using Chou-Fasman propensities.

        Uses a sliding window of length 7 to smooth per-residue predictions.

        Args:
            sequence: Amino acid sequence (single-letter codes).

        Returns:
            SecondaryStructurePrediction with keys:
                - ``ss_prediction``: string of ``H`` (helix), ``E`` (sheet),
                  ``C`` (coil), one per residue.
                - ``confidence_scores``: list of floats in [0, 1].
        """
        seq = sequence.upper()
        n = len(seq)
        if n == 0:
            return SecondaryStructurePrediction(
                ss_prediction="", confidence_scores=[]
            )

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
            labels = ["H", "E", "C"]

            sorted_props = sorted(props, reverse=True)
            best = max(range(3), key=lambda k: props[k])
            second_best = sorted_props[1] if len(sorted_props) > 1 else 0.0

            confidence = min(
                1.0, max(0.0, sorted_props[0] - second_best) / max(window, 1)
            )

            ss_chars.append(labels[best])
            confs.append(round(confidence, 4))

        ss_prediction = "".join(ss_chars)

        return SecondaryStructurePrediction(
            ss_prediction=ss_prediction,
            confidence_scores=confs,
        )
