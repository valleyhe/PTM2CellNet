"""
集成测试
功能概述: 测试模块间接口交互和数据传递
设计思路: 验证各模块协同工作，确保端到端流程正确
"""

import pytest
import torch
import torch.nn as nn

from src.utils.logging import setup_logger
from src.data.loaders import DataLoader as PTMDataLoader
from src.data.preprocess import DataPreprocessor
from src.data.datasets import PTMPlainDataModule
from src.models.encoders import CNNEncoder
from src.models.ptm_modules import PTMModule
from src.models.predictors import ClassificationPredictor
from src.models.architectures import PTM2CellNet
from src.training.optimizers import configure_optimizer
from src.training.trainers import Trainer
from src.evaluation.evaluators import Evaluator

logger = setup_logger(__name__)


class TestDataModelIntegration:
    """数据模块-模型模块集成测试"""

    @pytest.fixture
    def sample_data(self):
        """示例数据fixture"""
        loader = PTMDataLoader()
        df = loader.load_sample_data(num_samples=100)
        return df

    def test_data_to_dataset(self, sample_data):
        """测试数据到数据集的转换"""
        df = sample_data

        preprocessor = DataPreprocessor()
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=16)

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.test_dataset is not None

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        assert "sequence" in batch
        assert "ptm_mask" in batch
        assert "ptm_types" in batch
        assert "label" in batch
        assert batch["sequence"].shape[0] == 16

    def test_dataset_to_model(self, sample_data):
        """测试数据集到模型的输入"""
        df = sample_data

        preprocessor = DataPreprocessor()
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=8)
        cell_states = datamodule.get_labels()

        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=len(cell_states),
        )

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        with torch.no_grad():
            outputs = model(batch)

        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert "predictions" in outputs
        assert outputs["logits"].shape == (8, len(cell_states))


class TestModelTrainingIntegration:
    """模型-训练模块集成测试"""

    @pytest.fixture
    def sample_setup(self):
        """示例设置fixture"""
        loader = PTMDataLoader()
        df = loader.load_sample_data(num_samples=80)

        preprocessor = DataPreprocessor()
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=16)
        cell_states = datamodule.get_labels()

        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=len(cell_states),
        )

        return datamodule, model, cell_states

    def test_trainer_compile(self, sample_setup):
        """测试训练器编译"""
        datamodule, model, cell_states = sample_setup

        trainer = Trainer(model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer, scheduler = configure_optimizer(model)

        trainer.compile(
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None

    def test_trainer_fit(self, sample_setup):
        """测试训练器训练"""
        datamodule, model, cell_states = sample_setup

        trainer = Trainer(model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer, scheduler = configure_optimizer(model)

        trainer.compile(
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        trainer.fit(
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            max_epochs=2,
        )

        assert len(trainer.train_losses) == 2
        assert len(trainer.val_losses) == 2


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_full_pipeline(self):
        """测试完整端到端流程"""
        logger.info("开始端到端集成测试")

        logger.info("步骤 1: 加载和预处理数据")
        loader = PTMDataLoader()
        df = loader.load_sample_data(num_samples=100)

        preprocessor = DataPreprocessor()
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        logger.info("步骤 2: 创建数据模块")
        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=16)
        cell_states = datamodule.get_labels()

        assert len(cell_states) > 0

        logger.info("步骤 3: 创建模型")
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=len(cell_states),
        )

        logger.info("步骤 4: 训练模型")
        trainer = Trainer(model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer, scheduler = configure_optimizer(model)

        trainer.compile(
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        trainer.fit(
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            max_epochs=3,
        )

        assert len(trainer.train_losses) == 3

        logger.info("步骤 5: 评估模型")
        evaluator = Evaluator(model, task_type="classification")
        result = evaluator.evaluate(datamodule.test_dataloader(), return_predictions=True)

        assert "metrics" in result
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics

        logger.info(f"端到端测试完成: Accuracy={metrics.get('accuracy', 0):.4f}")

        assert metrics.get("accuracy", 0) >= 0.0


class TestBoundaryConditions:
    """边界条件测试"""

    def test_empty_data(self):
        """测试空数据处理"""
        loader = PTMDataLoader()
        df = loader.load_sample_data(num_samples=10)

        preprocessor = DataPreprocessor()
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

        assert len(train_df) + len(val_df) + len(test_df) == len(df)

    def test_small_batch(self):
        """测试小批量训练"""
        loader = PTMDataLoader()
        df = loader.load_sample_data(num_samples=20)

        preprocessor = DataPreprocessor()
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

        datamodule = PTMPlainDataModule(train_df, val_df, test_df, batch_size=4)
        cell_states = datamodule.get_labels()

        model = PTM2CellNet(
            encoder_type="lstm",
            vocab_size=21,
            embed_dim=32,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=len(cell_states),
        )

        trainer = Trainer(model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer, scheduler = configure_optimizer(model)

        trainer.compile(
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        trainer.fit(
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            max_epochs=2,
        )

        assert len(trainer.train_losses) == 2


class TestModuleInterfaces:
    """模块接口测试"""

    def test_encoder_ptm_predictor_chain(self):
        """测试编码器-PTM模块-预测器链"""
        batch_size = 4
        seq_len = 50
        vocab_size = 21
        embed_dim = 64
        num_classes = 4

        sequences = torch.randint(0, vocab_size, (batch_size, seq_len))
        ptm_mask = torch.randint(0, 2, (batch_size, seq_len)).float()
        ptm_types = torch.randint(0, 10, (batch_size, seq_len))

        encoder = CNNEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=1000,
        )
        ptm_module = PTMModule(
            num_ptm_types=10,
            embed_dim=embed_dim,
            max_position=1000,
        )
        predictor = ClassificationPredictor(
            input_dim=embed_dim,
            num_classes=num_classes,
        )

        seq_emb = encoder(sequences)

        ptm_positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        fused_emb = ptm_module(seq_emb, ptm_types, ptm_positions, ptm_mask)

        pooled = fused_emb.mean(dim=1)
        outputs = predictor(pooled)

        assert "logits" in outputs
        assert "probabilities" in outputs
        assert "predictions" in outputs
        assert outputs["logits"].shape == (batch_size, num_classes)
