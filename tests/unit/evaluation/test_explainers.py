import pandas as pd
import torch

from src.evaluation.explainers import LeaveOnePTMOutScorer, aggregate_by_protein


class MockModel:
    def __call__(self, batch):
        ptm_count = int(batch["ptm_mask"].sum().item())
        positive = min(0.2 + 0.3 * ptm_count, 0.95)
        return {
            "probabilities": torch.tensor([[1 - positive, positive]], dtype=torch.float32),
            "predictions": torch.tensor([1 if positive >= 0.5 else 0], dtype=torch.long),
        }


class CaptureBatchModel:
    def __init__(self) -> None:
        self.last_batch = None

    def __call__(self, batch):
        self.last_batch = batch
        return {
            "probabilities": torch.tensor([[0.4, 0.6]], dtype=torch.float32),
            "predictions": torch.tensor([1], dtype=torch.long),
        }


def test_leave_one_ptm_out_ranks_larger_probability_drop_first() -> None:
    scorer = LeaveOnePTMOutScorer(target_class_index=1)
    row = pd.Series(
        {
            "sample_id": "S1",
            "protein_id": "P04637",
            "sequence": "ACDEFG",
            "ptm_sites": '[{"position": 2, "type": "phosphorylation"}, {"position": 5, "type": "acetylation"}]',
        }
    )

    ranked = scorer.rank_row(MockModel(), row)

    assert len(ranked) == 2
    assert ranked[0].delta_probability >= ranked[1].delta_probability
    assert ranked[0].baseline_probability > ranked[0].perturbed_probability


def test_leave_one_ptm_out_builds_sequence_and_ptm_tensors() -> None:
    scorer = LeaveOnePTMOutScorer(
        target_class_index=1,
        amino_acids="ACDEFGHIKLMNPQRSTVWY",
        ptm_type_to_index={"phosphorylation": 1, "acetylation": 2},
    )
    row = pd.Series(
        {
            "sample_id": "S1",
            "protein_id": "P04637",
            "sequence": "ACDEFG",
            "ptm_sites": '[{"position": 2, "type": "phosphorylation"}]',
        }
    )
    model = CaptureBatchModel()

    scorer.rank_row(model, row)

    assert model.last_batch is not None
    assert model.last_batch["sequence"].shape == (1, 6)
    assert model.last_batch["ptm_types"].shape == (1, 6)
    assert model.last_batch["ptm_positions"].shape == (1, 6)


def test_aggregate_by_protein_keeps_max_delta_per_protein() -> None:
    scorer = LeaveOnePTMOutScorer(target_class_index=1)
    row = pd.Series(
        {
            "sample_id": "S1",
            "protein_id": "P04637",
            "sequence": "ACDEFG",
            "ptm_sites": '[{"position": 2, "type": "phosphorylation"}, {"position": 5, "type": "acetylation"}]',
        }
    )

    ranked = scorer.rank_row(MockModel(), row)
    scores = aggregate_by_protein(ranked)

    assert list(scores.keys()) == ["P04637"]
    assert scores["P04637"] == max(candidate.delta_probability for candidate in ranked)
