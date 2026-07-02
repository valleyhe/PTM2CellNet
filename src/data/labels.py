"""Shared label mapping utilities for training scripts."""

from typing import Dict, List, Tuple

import pandas as pd


def derive_label_mapping(
    df: pd.DataFrame, label_col: str = "cell_state"
) -> Tuple[List[str], Dict[str, int]]:
    """Derive label mapping from a DataFrame with a label column.

    Args:
        df: DataFrame containing a label column.
        label_col: Name of the label column.

    Returns:
        (cell_states, label_to_idx) tuple where cell_states is sorted unique
        labels and label_to_idx maps each label to its index.

    Raises:
        ValueError: If fewer than 2 classes are found.
    """
    labels = sorted(df[label_col].dropna().astype(str).unique())
    if len(labels) < 2:
        raise ValueError(
            f"需要至少 2 个类别才能训练，当前只有 {len(labels)} 个类别: {labels}"
        )
    label_to_idx = {label: i for i, label in enumerate(labels)}
    return labels, label_to_idx


__all__ = ["derive_label_mapping"]
