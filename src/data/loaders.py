"""
数据加载器模块
功能概述: 从各种数据源加载蛋白质序列和PTM数据
设计思路: 提供统一的加载接口，支持CSV、FASTA、JSON等多种格式
"""

import os
from typing import Any, Dict, List, Optional
import json

import pandas as pd
from Bio import SeqIO

from ..utils.logging import setup_logger
from ..utils.helpers import validate_sequence, clean_sequence

logger = setup_logger(__name__)


class DataLoader:
    """
    数据加载器类
    负责从各种数据源加载原始数据
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据加载器

        参数:
            config: 配置字典，包含路径等设置
        """
        self.config = config or {}
        self.data_raw_dir = self.config.get("paths", {}).get("data_raw", "data/raw")
        self.valid_amino_acids = set(
            self.config.get("data", {}).get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY")
        )

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        从CSV文件加载数据

        参数:
            file_path: CSV文件路径

        返回:
            DataFrame对象

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV文件不存在: {file_path}")

        logger.info("从CSV加载数据: %s", file_path)
        df = pd.read_csv(file_path)
        logger.info("加载完成，共 %d 条记录", len(df))
        return df

    def load_from_fasta(self, file_path: str) -> Dict[str, str]:
        """
        从FASTA文件加载蛋白质序列

        参数:
            file_path: FASTA文件路径

        返回:
            字典，键为序列ID，值为序列字符串

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FASTA文件不存在: {file_path}")

        logger.info("从FASTA加载序列: %s", file_path)
        sequences = {}

        for record in SeqIO.parse(file_path, "fasta"):
            seq_id = record.id
            sequence = clean_sequence(str(record.seq))
            is_valid, error_msg = validate_sequence(sequence, valid_amino_acids=self.valid_amino_acids)
            if is_valid:
                sequences[seq_id] = sequence
            else:
                logger.warning("跳过无效序列 %s: %s", seq_id, error_msg)

        logger.info("加载完成，共 %d 条有效序列", len(sequences))
        return sequences

    def load_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从JSON文件加载数据

        参数:
            file_path: JSON文件路径

        返回:
            数据列表

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON文件不存在: {file_path}")

        logger.info("从JSON加载数据: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise TypeError("JSON数组中的每一项都必须是对象")
            records = data
        else:
            raise TypeError("JSON内容必须是对象或对象数组")

        logger.info("加载完成，共 %d 条记录", len(records))
        return records

    def load_sample_data(self, num_samples: int = 100) -> pd.DataFrame:
        """
        加载示例数据（用于测试）

        参数:
            num_samples: 样本数量

        返回:
            示例数据DataFrame
        """
        logger.info("生成示例数据，共 %d 个样本", num_samples)

        import random

        valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        ptm_types = ["phosphorylation", "acetylation", "methylation", "ubiquitination"]
        cell_states = ["proliferation", "differentiation", "apoptosis", "quiescence"]

        data = []
        for i in range(num_samples):
            seq_length = random.randint(50, 500)
            sequence = "".join(random.choice(valid_amino_acids) for _ in range(seq_length))

            num_ptms = random.randint(0, 5)
            ptm_sites = []
            positions = random.sample(range(1, seq_length + 1), min(num_ptms, seq_length))
            for pos in positions:
                ptm_sites.append({
                    "position": pos,
                    "type": random.choice(ptm_types),
                    "amino_acid": sequence[pos - 1]
                })

            data.append({
                "id": f"sample_{i:04d}",
                "sequence": sequence,
                "ptm_sites": json.dumps(ptm_sites),
                "cell_state": random.choice(cell_states)
            })

        df = pd.DataFrame(data)
        logger.info("示例数据生成完成")
        return df

    def load_combined_data(
        self,
        sequence_file: Optional[str] = None,
        ptm_file: Optional[str] = None,
        label_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        从多个文件加载并合并数据

        参数:
            sequence_file: 序列文件路径
            ptm_file: PTM数据文件路径
            label_file: 标签文件路径

        返回:
            合并后的DataFrame
        """
        data_frames = []

        if sequence_file:
            if sequence_file.endswith(".fasta") or sequence_file.endswith(".fa"):
                seq_dict = self.load_from_fasta(sequence_file)
                seq_df = pd.DataFrame(list(seq_dict.items()), columns=["id", "sequence"])
                data_frames.append(seq_df)
            elif sequence_file.endswith(".csv"):
                data_frames.append(self.load_from_csv(sequence_file))

        if ptm_file and ptm_file.endswith(".csv"):
            data_frames.append(self.load_from_csv(ptm_file))

        if label_file and label_file.endswith(".csv"):
            data_frames.append(self.load_from_csv(label_file))

        if not data_frames:
            logger.warning("未提供任何数据文件")
            return pd.DataFrame()

        if len(data_frames) == 1:
            return data_frames[0]

        result = data_frames[0]
        for df in data_frames[1:]:
            common_cols = [col for col in result.columns if col in df.columns]
            if common_cols:
                result = result.merge(df, on=common_cols, how="outer")
            else:
                result = pd.concat([result, df], axis=1)

        logger.info("数据合并完成，共 %d 条记录", len(result))
        return result
