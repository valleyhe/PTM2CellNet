"""Behavior snapshot tests for scripts/data_statistics.py (N03)."""

import json

import pandas as pd
import pytest

import scripts.data_statistics as ds


def _df_with_sites(ptm_sites_list, cell_states=None, sequences=None):
    n = len(ptm_sites_list)
    return pd.DataFrame(
        {
            "sequence": sequences or ["ACDEFGHIK" * (i + 1) for i in range(n)],
            "cell_state": cell_states or ["proliferation"] * n,
            "ptm_sites": [json.dumps(s) for s in ptm_sites_list],
        }
    )


class TestAnalyzeDataframe:
    def test_basic_statistics(self):
        df = _df_with_sites(
            [
                [{"position": 5, "type": "phosphorylation"}],
                [{"position": 3, "type": "acetylation"}, {"position": 9, "type": "phosphorylation"}],
                [],
            ],
            cell_states=["proliferation", "differentiation", "proliferation"],
        )
        stats = ds.analyze_dataframe(df, "train")

        assert stats["样本总数"] == 3
        assert stats["序列长度"]["最小值"] == 9
        assert stats["序列长度"]["最大值"] == 27
        assert stats["序列长度"]["平均值"] == pytest.approx(18.0)
        assert stats["细胞状态分布"] == {"proliferation": 2, "differentiation": 1}
        assert stats["PTM位点数量"]["最小值"] == 0
        assert stats["PTM位点数量"]["最大值"] == 2
        # PTM 类型分布聚合
        assert stats["PTM类型分布"] == {
            "phosphorylation": 2,
            "acetylation": 1,
        }

    def test_ptm_sites_accepted_as_objects(self):
        # 非字符串（dict 列表）也应工作
        df = pd.DataFrame(
            {
                "sequence": ["ACDEFG"],
                "cell_state": ["proliferation"],
                "ptm_sites": [[{"position": 2, "type": "methylation"}]],
            }
        )
        stats = ds.analyze_dataframe(df, "x")
        assert stats["PTM位点数量"]["平均值"] == 1
        assert stats["PTM类型分布"] == {"methylation": 1}

    def test_malformed_json_counts_as_zero(self):
        df = pd.DataFrame(
            {
                "sequence": ["ACDEFG"],
                "cell_state": ["proliferation"],
                "ptm_sites": ["{not valid json"],
            }
        )
        stats = ds.analyze_dataframe(df, "x")
        assert stats["PTM位点数量"]["最小值"] == 0
        # 无有效 PTM 类型时不输出分布键（行为快照）
        assert "PTM类型分布" not in stats

    def test_median_handles_empty_ptm_counts(self):
        df = _df_with_sites([[], [], []])
        stats = ds.analyze_dataframe(df, "x")
        assert stats["PTM位点数量"]["中位数"] == 0
        assert stats["PTM位点数量"]["平均值"] == 0.0
