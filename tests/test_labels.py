"""
标签映射工具的单元测试

覆盖 derive_label_mapping 的关键映射逻辑与未知/异常输入处理。
"""

import numpy as np
import pandas as pd
import pytest

from src.data.labels import derive_label_mapping


class TestDeriveLabelMapping:
    """derive_label_mapping 映射逻辑测试。"""

    def test_distinct_labels_sorted_with_zero_based_indices(self):
        df = pd.DataFrame({"cell_state": ["B_cell", "T_cell", "NK_cell"]})

        cell_states, label_to_idx = derive_label_mapping(df)

        assert cell_states == sorted(["B_cell", "T_cell", "NK_cell"])
        assert label_to_idx == {
            cell_states[0]: 0,
            cell_states[1]: 1,
            cell_states[2]: 2,
        }

    def test_key_ptm_related_states_map_to_consecutive_indices(self):
        # 关键细胞状态 / PTM 生物学语境下的典型标签
        df = pd.DataFrame({"cell_state": ["phosphorylation", "ubiquitination", "acetylation"]})

        cell_states, label_to_idx = derive_label_mapping(df)

        assert set(cell_states) == {
            "phosphorylation",
            "ubiquitination",
            "acetylation",
        }
        # 索引必须连续且唯一，覆盖 0..N-1
        assert sorted(label_to_idx.values()) == list(range(len(cell_states)))
        for label, idx in label_to_idx.items():
            assert cell_states[idx] == label

    def test_duplicate_labels_collapse_to_unique(self):
        df = pd.DataFrame({"cell_state": ["A", "A", "B", "B", "C"]})

        cell_states, label_to_idx = derive_label_mapping(df)

        assert cell_states == ["A", "B", "C"]
        assert label_to_idx == {"A": 0, "B": 1, "C": 2}

    def test_nan_values_are_dropped(self):
        df = pd.DataFrame({"cell_state": ["A", None, "B", np.nan]})

        cell_states, label_to_idx = derive_label_mapping(df)

        assert cell_states == ["A", "B"]
        assert label_to_idx == {"A": 0, "B": 1}

    def test_custom_label_column(self):
        df = pd.DataFrame({"ptm_type": ["kinase", "phosphatase"]})

        cell_states, label_to_idx = derive_label_mapping(df, label_col="ptm_type")

        assert cell_states == ["kinase", "phosphatase"]
        assert label_to_idx == {"kinase": 0, "phosphatase": 1}

    def test_numeric_labels_are_stringified(self):
        df = pd.DataFrame({"cell_state": [2, 10, 1]})

        cell_states, label_to_idx = derive_label_mapping(df)

        # 排序基于字符串形式（"1" < "10" < "2"）
        assert cell_states == ["1", "10", "2"]
        assert label_to_idx == {"1": 0, "10": 1, "2": 2}


class TestDeriveLabelMappingInvalidInput:
    """未知/异常输入处理测试。"""

    def test_single_class_raises_value_error(self):
        df = pd.DataFrame({"cell_state": ["only_one", "only_one", "only_one"]})

        with pytest.raises(ValueError, match="至少 2 个类别"):
            derive_label_mapping(df)

    def test_empty_column_raises_value_error(self):
        df = pd.DataFrame({"cell_state": pd.Series([], dtype=object)})

        with pytest.raises(ValueError, match="至少 2 个类别"):
            derive_label_mapping(df)

    def test_all_nan_column_raises_value_error(self):
        df = pd.DataFrame({"cell_state": [None, None, np.nan]})

        with pytest.raises(ValueError, match="至少 2 个类别"):
            derive_label_mapping(df)

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({"other_col": ["A", "B"]})

        with pytest.raises(KeyError):
            derive_label_mapping(df, label_col="cell_state")
