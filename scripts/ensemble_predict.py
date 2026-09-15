#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型集成预测脚本
功能: 整合多个模型的预测结果
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.io import safe_torch_load
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelEnsemble:
    """模型集成器"""

    def __init__(
        self,
        model_configs: List[Dict],
        device: str = "cpu",
    ):
        """
        初始化集成器

        参数:
            model_configs: 模型配置列表
                [{'path': 'model.pt', 'type': 'cnn_multitask', 'weight': 1.0}, ...]
            device: 计算设备
        """
        self.device = torch.device(device)
        self.model_configs = model_configs
        self.models = []
        self.weights = []

        self._load_models()

    def _load_models(self):
        """加载所有模型"""
        for config in self.model_configs:
            model_path = config["path"]
            model_type = config.get("type", "cnn_multitask")
            weight = config.get("weight", 1.0)

            logger.info(f"加载模型: {model_path} (类型: {model_type}, 权重: {weight})")

            if model_type == "cnn_multitask":
                model = self._load_cnn_model(model_path)
            elif model_type.startswith("esm2"):
                model = self._load_esm2_model(model_path, model_type)
            else:
                logger.warning(f"未知模型类型: {model_type}")
                continue

            model.eval()
            self.models.append(model)
            self.weights.append(weight)

        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        logger.info(f"已加载 {len(self.models)} 个模型")

    def _load_cnn_model(self, path):
        """加载CNN模型"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(encoder_type="cnn")
        state_dict = safe_torch_load(path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def _load_esm2_model(self, path, model_type):
        """加载ESM-2模型"""
        import esm

        if "t12" in model_type:
            esm_name = "esm2_t12_35M_UR50D"
        else:
            esm_name = "esm2_t6_8M_UR50D"

        esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_name)

        class ESM2Wrapper(nn.Module):
            def __init__(self, esm_model, state_dict):
                super().__init__()
                self.esm = esm_model
                self.embed_dim = esm_model.embed_dim
                self.classifier = nn.Sequential(
                    nn.Linear(self.embed_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2),
                )
                self.load_state_dict(state_dict, strict=False)
                self.alphabet = alphabet
                self.batch_converter = alphabet.get_batch_converter()

            def forward(self, sequences, ptm_type=None):
                # sequences can be a list of strings or a batch tensor
                if isinstance(sequences, (list, tuple)):
                    # Convert sequences to ESM format: list of (label, seq) tuples
                    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                    batch_tokens = batch_tokens.to(next(self.esm.parameters()).device)

                    with torch.no_grad():
                        results = self.esm(batch_tokens, repr_layers=[self.esm.num_layers])
                        token_representations = results["representations"][self.esm.num_layers]

                    # Mean pooling over sequence (exclude BOS/EOS tokens)
                    mask = batch_tokens.ne(self.alphabet.padding_idx)
                    # Remove BOS (index 0)
                    mask[:, 0] = False

                    # For each sequence, find the EOS position and exclude it
                    for i in range(mask.shape[0]):
                        eos_idx = (batch_tokens[i] == self.alphabet.eos_idx).nonzero(as_tuple=False)
                        if len(eos_idx) > 0:
                            mask[i, eos_idx[0, 0]] = False

                    # Mean pool
                    masked_repr = token_representations * mask.unsqueeze(-1)
                    lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    pooled = masked_repr.sum(dim=1) / lengths
                else:
                    # Already a tensor — pass through ESM directly
                    pooled = sequences

                logits = self.classifier(pooled)
                probs = torch.softmax(logits, dim=-1)
                return {"probs": probs, "logits": logits}

        state_dict = safe_torch_load(path, map_location=self.device)
        model = ESM2Wrapper(esm_model, state_dict)
        model.to(self.device)
        return model

    def predict_voting(
        self,
        sequences: List[str],
        ptm_type: str = "Phosphorylation",
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        投票法集成预测

        参数:
            sequences: 序列列表
            ptm_type: PTM类型
            threshold: 分类阈值

        返回:
            预测概率
        """
        all_probs = []

        with torch.no_grad():
            for model, weight in zip(self.models, self.weights, strict=False):
                # 根据模型类型进行预测
                if hasattr(model, "forward"):
                    try:
                        output = model(sequences, ptm_type)
                        probs = output["probs"][:, 1].cpu().numpy()
                        all_probs.append(probs * weight)
                    except Exception as e:
                        logger.warning(f"模型预测失败: {e}")

        if not all_probs:
            return np.zeros(len(sequences))

        # 加权平均
        ensemble_probs = np.sum(all_probs, axis=0)
        return ensemble_probs

    def predict_averaging(
        self,
        sequences: List[str],
        ptm_type: str = "Phosphorylation",
    ) -> np.ndarray:
        """
        平均法集成预测

        参数:
            sequences: 序列列表
            ptm_type: PTM类型

        返回:
            平均预测概率
        """
        return self.predict_voting(sequences, ptm_type)

    def predict_stacking(
        self,
        sequences: List[str],
        ptm_type: str = "Phosphorylation",
        meta_weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Stacking集成预测

        参数:
            sequences: 序列列表
            ptm_type: PTM类型
            meta_weights: 元学习器权重

        返回:
            集成预测概率
        """
        if meta_weights:
            original_weights = self.weights
            self.weights = [w / sum(meta_weights) for w in meta_weights]
            result = self.predict_voting(sequences, ptm_type)
            self.weights = original_weights
            return result

        return self.predict_voting(sequences, ptm_type)


def evaluate_ensemble(
    ensemble: ModelEnsemble,
    test_data: pd.DataFrame,
    ptm_type: str = "Phosphorylation",
) -> Dict:
    """
    评估集成模型

    参数:
        ensemble: 集成模型
        test_data: 测试数据
        ptm_type: PTM类型

    返回:
        评估指标
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

    sequences = test_data["sequence_window"].tolist()
    labels = test_data["label"].values

    # 预测
    probs = ensemble.predict_averaging(sequences, ptm_type)
    preds = (probs > 0.5).astype(int)

    # 计算指标
    metrics = {
        "auroc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5,
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }

    return metrics


def compare_models(
    model_configs: List[Dict],
    test_data: pd.DataFrame,
    ptm_type: str = "Phosphorylation",
) -> pd.DataFrame:
    """
    比较单个模型与集成模型的性能

    参数:
        model_configs: 模型配置列表
        test_data: 测试数据
        ptm_type: PTM类型

    返回:
        比较结果DataFrame
    """
    from sklearn.metrics import roc_auc_score

    results = []
    sequences = test_data["sequence_window"].tolist()
    labels = test_data["label"].values

    # 单模型评估
    for config in model_configs:
        ensemble = ModelEnsemble([config])
        probs = ensemble.predict_averaging(sequences, ptm_type)
        auroc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5

        results.append(
            {
                "model": config["path"],
                "type": config.get("type", "unknown"),
                "auroc": auroc,
                "method": "single",
            }
        )

    # 集成评估
    ensemble = ModelEnsemble(model_configs)
    probs = ensemble.predict_averaging(sequences, ptm_type)
    auroc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5

    results.append(
        {
            "model": "ensemble",
            "type": "ensemble",
            "auroc": auroc,
            "method": "averaging",
        }
    )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="模型集成预测")

    parser.add_argument("--models", nargs="+", required=True, help="模型路径列表")
    parser.add_argument("--types", nargs="+", help="模型类型列表")
    parser.add_argument("--weights", nargs="+", type=float, help="模型权重列表")
    parser.add_argument("--test-data", required=True, help="测试数据文件")
    parser.add_argument("--ptm-type", default="Phosphorylation", help="PTM类型")
    parser.add_argument("--output", default="ensemble_results.csv", help="输出文件")
    parser.add_argument("--device", default="cpu", help="计算设备")

    args = parser.parse_args()

    # 构建模型配置
    model_configs = []
    for i, model_path in enumerate(args.models):
        config = {"path": model_path}

        if args.types and i < len(args.types):
            config["type"] = args.types[i]
        else:
            config["type"] = "cnn_multitask"

        if args.weights and i < len(args.weights):
            config["weight"] = args.weights[i]
        else:
            config["weight"] = 1.0

        model_configs.append(config)

    # 加载测试数据
    test_data = pd.read_csv(args.test_data)
    logger.info(f"加载测试数据: {len(test_data)} 样本")

    # 创建集成模型
    ensemble = ModelEnsemble(model_configs, args.device)

    # 评估
    metrics = evaluate_ensemble(ensemble, test_data, args.ptm_type)

    logger.info("=" * 50)
    logger.info("集成模型评估结果:")
    logger.info(f"  AUROC: {metrics['auroc']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1: {metrics['f1']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")

    # 模型比较
    comparison = compare_models(model_configs, test_data, args.ptm_type)
    logger.info("\n模型比较:")
    print(comparison.to_string(index=False))

    # 保存结果
    comparison.to_csv(args.output, index=False)
    logger.info(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
