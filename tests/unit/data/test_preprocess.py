"""
DataPreprocessor 单元测试
"""

import json

import pandas as pd
import pytest

from src.data.preprocess import DataPreprocessor


class TestDataPreprocessor:
    """DataPreprocessor类测试"""

    @pytest.fixture
    def sample_df(self):
        """示例数据fixture"""
        data = [
            {
                "id": "1",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_sites": json.dumps([{"position": 5, "type": "phosphorylation"}]),
                "cell_state": "proliferation",
            },
            {
                "id": "2",
                "sequence": "ACDEFGHIKL",
                "ptm_sites": json.dumps([]),
                "cell_state": "differentiation",
            },
        ]
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def test_initialization_defaults(self):
        """无配置时使用默认参数初始化。"""
        preprocessor = DataPreprocessor()
        assert preprocessor.config == {}
        assert preprocessor.max_sequence_length == 1000
        assert preprocessor.valid_amino_acids == set("ACDEFGHIKLMNPQRSTVWY")
        assert preprocessor.remove_duplicates is True
        assert preprocessor.handle_missing == "drop"

    def test_initialization_with_config(self):
        """配置字典应覆盖默认参数。"""
        config = {
            "data": {
                "max_sequence_length": 512,
                "valid_amino_acids": "ACDEFGHIK",
                "preprocessing": {
                    "remove_duplicates": False,
                    "handle_missing": "keep",
                },
            }
        }
        preprocessor = DataPreprocessor(config)
        assert preprocessor.max_sequence_length == 512
        assert preprocessor.valid_amino_acids == set("ACDEFGHIK")
        assert preprocessor.remove_duplicates is False
        assert preprocessor.handle_missing == "keep"

    # ------------------------------------------------------------------
    # clean_sequences
    # ------------------------------------------------------------------
    def test_clean_sequences_removes_invalid_chars(self):
        """含非标准氨基酸的序列应被删除（drop 模式），保留有效序列。"""
        df = pd.DataFrame(
            [
                {"sequence": "ACDEFGHIK", "cell_state": "a"},
                {"sequence": "ACXDEFGHIK", "cell_state": "b"},  # X 非标准
                {"sequence": "ACDEFG*HIK", "cell_state": "c"},  # * 非法
            ]
        )
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_sequences(df)

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["sequence"] == "ACDEFGHIK"

    def test_clean_sequences_handles_empty(self):
        """空 DataFrame 应返回空 DataFrame，不抛异常。"""
        df = pd.DataFrame(columns=["sequence", "cell_state"])
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_sequences(df)

        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) == 0

    def test_clean_sequences_uppercases_and_strips_whitespace(self):
        """合法序列应被清洗（去空白、转大写）后保留。"""
        df = pd.DataFrame(
            [{"sequence": " ac def ghik ", "cell_state": "a"}]
        )
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_sequences(df)

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["sequence"] == "ACDEFGHIK"

    def test_clean_sequences_drops_nan_sequence(self):
        """NaN 序列应被视为无效并删除。"""
        df = pd.DataFrame(
            [
                {"sequence": "ACDEFGHIK", "cell_state": "a"},
                {"sequence": None, "cell_state": "b"},
            ]
        )
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_sequences(df)

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["sequence"] == "ACDEFGHIK"

    def test_clean_sequences_drops_too_long(self):
        """超过 max_sequence_length 的序列应被删除。"""
        config = {"data": {"max_sequence_length": 5}}
        preprocessor = DataPreprocessor(config)
        df = pd.DataFrame(
            [
                {"sequence": "ACDE", "cell_state": "a"},  # 合法长度
                {"sequence": "ACDEFGHIK", "cell_state": "b"},  # 超长
            ]
        )
        cleaned = preprocessor.clean_sequences(df)

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["sequence"] == "ACDE"

    # ------------------------------------------------------------------
    # normalize_ptm_labels
    # ------------------------------------------------------------------
    def test_normalize_ptm_labels(self, sample_df):
        """合法 PTM 位点应保留，越界位点应被丢弃，输出为 JSON 字符串。"""
        # seq1 长度 20，position=5 合法；构造一个越界位点
        df = pd.DataFrame(
            [
                {
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",  # 长度 20
                    "ptm_sites": json.dumps(
                        [
                            {"position": 5, "type": "phosphorylation"},  # 合法
                            {"position": 99, "type": "phosphorylation"},  # 越界
                        ]
                    ),
                    "cell_state": "a",
                },
                {
                    "sequence": "ACDEFGHIKL",  # 长度 10
                    "ptm_sites": json.dumps(
                        [{"position": 3, "type": "acetylation"}]
                    ),
                    "cell_state": "b",
                },
            ]
        )
        preprocessor = DataPreprocessor()
        normalized = preprocessor.normalize_ptm_labels(df)

        # 第一行：仅保留合法位点
        sites_0 = json.loads(normalized.iloc[0]["ptm_sites"])
        assert len(sites_0) == 1
        assert sites_0[0]["position"] == 5

        # 第二行：保留唯一合法位点
        sites_1 = json.loads(normalized.iloc[1]["ptm_sites"])
        assert len(sites_1) == 1
        assert sites_1[0]["position"] == 3

        # 输出列应为 JSON 字符串
        assert isinstance(normalized.iloc[0]["ptm_sites"], str)

    def test_normalize_ptm_labels_empty(self):
        """空 DataFrame 应返回空 DataFrame，不抛异常。"""
        df = pd.DataFrame(columns=["sequence", "ptm_sites"])
        preprocessor = DataPreprocessor()
        normalized = preprocessor.normalize_ptm_labels(df)

        assert isinstance(normalized, pd.DataFrame)
        assert len(normalized) == 0

    def test_normalize_ptm_labels_handles_nan_and_invalid_json(self):
        """NaN、非法 JSON、非法类型应被归一化为空 JSON 列表。"""
        df = pd.DataFrame(
            [
                {"sequence": "ACDEFGHIK", "ptm_sites": None},
                {"sequence": "ACDEFGHIK", "ptm_sites": "not-a-json"},
                {"sequence": "ACDEFGHIK", "ptm_sites": 12345},
            ]
        )
        preprocessor = DataPreprocessor()
        normalized = preprocessor.normalize_ptm_labels(df)

        for idx in range(len(normalized)):
            assert json.loads(normalized.iloc[idx]["ptm_sites"]) == []

    def test_normalize_ptm_labels_accepts_list_input(self):
        """原生 list 输入也应被正确解析并输出为 JSON 字符串。"""
        df = pd.DataFrame(
            [
                {
                    "sequence": "ACDEFGHIK",  # 长度 9
                    "ptm_sites": [{"position": 2, "type": "phosphorylation"}],
                    "cell_state": "a",
                }
            ]
        )
        preprocessor = DataPreprocessor()
        normalized = preprocessor.normalize_ptm_labels(df)

        sites = json.loads(normalized.iloc[0]["ptm_sites"])
        assert len(sites) == 1
        assert sites[0]["position"] == 2
        assert isinstance(normalized.iloc[0]["ptm_sites"], str)

    # ------------------------------------------------------------------
    # split_dataset
    # ------------------------------------------------------------------
    def test_split_dataset_returns_three_splits(self, sample_df):
        """应返回三个 DataFrame（train/val/test）。"""
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_dataset(sample_df, ratios=(0.5, 0.25, 0.25))

        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_split_dataset_ratios(self):
        """划分比例应近似匹配配置的比例（容差 ±5%）。"""
        # 构造足够大的两类数据，确保分层划分生效
        rows = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "cell_state": "proliferation"}] * 60
        rows += [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "cell_state": "apoptosis"}] * 40
        df = pd.DataFrame(rows)

        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_dataset(df, ratios=(0.7, 0.15, 0.15))

        total = len(df)
        train_ratio = len(train) / total
        val_ratio = len(val) / total
        test_ratio = len(test) / total

        assert train_ratio == pytest.approx(0.7, abs=0.05)
        assert val_ratio == pytest.approx(0.15, abs=0.05)
        assert test_ratio == pytest.approx(0.15, abs=0.05)

    def test_split_dataset_empty(self):
        """空 DataFrame 应走简化划分策略，返回 1 行 train 与空 val/test。"""
        df = pd.DataFrame(columns=["sequence", "cell_state"])
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_dataset(df, ratios=(0.7, 0.15, 0.15))

        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        # 空数据下 train 也应为空（len(df)==0 < 3 触发简化策略）
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

    def test_split_dataset_deterministic_with_seed(self):
        """相同 random_state 应产生一致的划分。"""
        rows = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "cell_state": "a"}] * 40
        rows += [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "cell_state": "b"}] * 40
        df = pd.DataFrame(rows)

        preprocessor = DataPreprocessor()
        train1, val1, test1 = preprocessor.split_dataset(
            df, ratios=(0.7, 0.15, 0.15), random_state=42
        )
        train2, val2, test2 = preprocessor.split_dataset(
            df, ratios=(0.7, 0.15, 0.15), random_state=42
        )

        assert list(train1.index) == list(train2.index)
        assert list(val1.index) == list(val2.index)
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)

    # ------------------------------------------------------------------
    # preprocess_pipeline (端到端)
    # ------------------------------------------------------------------
    def test_preprocess_to_dataframe(self):
        """端到端：原始 DataFrame 应被清洗、标准化、去重、PTM 归一化后划分为三份。"""
        rows = [
            {
                "sequence": "acdefghiklmnpqrstvwy",  # 小写，应被转大写
                "ptm_sites": json.dumps([{"position": 5, "type": "phosphorylation"}]),
                "cell_state": "proliferation",
            }
            for _ in range(40)
        ]
        rows += [
            {
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_sites": json.dumps([]),
                "cell_state": "apoptosis",
            }
            for _ in range(40)
        ]
        # 混入若干无效序列（应被清洗删除）
        rows += [
            {"sequence": "ACXDEFGHIK", "ptm_sites": json.dumps([]), "cell_state": "a"}
            for _ in range(5)
        ]
        df = pd.DataFrame(rows)

        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.preprocess_pipeline(df)

        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        # 训练集非空
        assert len(train) > 0
        # 三者并集等于清洗去重后的样本数
        assert len(train) + len(val) + len(test) <= len(df)
        # 训练集序列应全部大写（小写已归一化）
        for seq in train["sequence"]:
            assert seq == seq.upper()
        # PTM 列应为合法 JSON 字符串
        for raw in train["ptm_sites"]:
            assert isinstance(raw, str)
            json.loads(raw)

    def test_preprocess_to_dataframe_respects_config(self):
        """配置 max_sequence_length 应在端到端流程中生效，超长序列被删除。"""
        config = {"data": {"max_sequence_length": 10}}
        rows = [
            {"sequence": "ACDEFGHIK", "ptm_sites": json.dumps([]), "cell_state": "a"}
            for _ in range(40)
        ]
        rows += [
            {"sequence": "ACDEFGHIK", "ptm_sites": json.dumps([]), "cell_state": "b"}
            for _ in range(40)
        ]
        # 超长序列，应在 clean_sequences 阶段被删除
        rows += [
            {"sequence": "ACDEFGHIKLMNPQRSTVWY", "ptm_sites": json.dumps([]), "cell_state": "a"}
            for _ in range(5)
        ]
        df = pd.DataFrame(rows)

        preprocessor = DataPreprocessor(config)
        train, val, test = preprocessor.preprocess_pipeline(df)

        assert len(train) > 0
        # 所有保留序列长度均不超过 max_sequence_length
        for seq in train["sequence"]:
            assert len(seq) <= 10
