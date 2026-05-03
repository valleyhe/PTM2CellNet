"""
Lightning训练流程集成测试
测试端到端训练流程
"""

import os
import tempfile
import pytest
import torch
import lightning as L
import pandas as pd

from src.models.architectures import PTM2CellNet
from src.training.lightning_module import PTM2CellNetLightning
from src.data.lightning_datamodule import PTMDataModule
from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor


class TestLightningPipeline:
    """Lightning训练流程集成测试"""

    @pytest.fixture
    def sample_config(self):
        """示例配置"""
        return {
            "model": {
                "encoder_type": "transformer",
                "num_classes": 4,
                "hidden_dim": 64,
                "num_layers": 1,
                "num_heads": 2,
                "dropout": 0.1,
            },
            "training": {
                "max_epochs": 2,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "optimizer": "adamw",
                "num_workers": 0,
            },
            "data": {
                "max_sequence_length": 50,
                "num_workers": 0,
            },
        }

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        # 创建足够多的数据，确保预处理后有足够的训练样本
        sequences = []
        ptm_sites = []
        cell_states = []

        # 生成20个不同的样本
        for i in range(20):
            # 创建不同的序列
            seq = "ACDEFGHIKLMNPQRSTVWY" * 2 + "ACDEFGHIKLMN"[:i]
            sequences.append(seq)
            ptm_sites.append(f'[{{"position": {(i % 10) + 1}, "type": "phosphorylation"}}]')
            cell_states.append(["proliferation", "differentiation", "apoptosis", "quiescence"][i % 4])

        data = {
            "sequence": sequences,
            "ptm_sites": ptm_sites,
            "cell_state": cell_states,
        }
        return pd.DataFrame(data)

    def test_data_module_creation(self, sample_data, sample_config):
        """测试数据模块创建"""
        # 预处理数据
        preprocessor = DataPreprocessor(sample_config)
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(sample_data)

        # 创建数据模块
        datamodule = PTMDataModule(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=sample_config,
        )

        datamodule.setup()

        # 验证数据加载器
        train_loader = datamodule.train_dataloader()
        assert train_loader is not None

        batch = next(iter(train_loader))
        assert "sequence" in batch
        assert "label" in batch

    def test_model_creation(self, sample_config):
        """测试模型创建"""
        model = PTM2CellNet.from_config(sample_config)
        lightning_model = PTM2CellNetLightning(model, sample_config)

        assert lightning_model is not None
        assert lightning_model.learning_rate == sample_config["training"]["learning_rate"]

    def test_training_step(self, sample_config, sample_data):
        """测试训练步骤"""
        # 准备数据
        preprocessor = DataPreprocessor(sample_config)
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(sample_data)

        datamodule = PTMDataModule(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=sample_config,
        )
        datamodule.setup()

        # 创建模型
        model = PTM2CellNet.from_config(sample_config)
        lightning_model = PTM2CellNetLightning(model, sample_config)

        # 获取批次
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # 执行训练步骤
        loss = lightning_model.training_step(batch, 0)

        assert loss is not None
        assert loss.item() > 0

    def test_end_to_end_training(self, sample_config, sample_data):
        """测试端到端训练流程"""
        # 准备数据
        preprocessor = DataPreprocessor(sample_config)
        train_df, val_df, test_df = preprocessor.preprocess_pipeline(sample_data)

        datamodule = PTMDataModule(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=sample_config,
        )

        # 创建模型
        model = PTM2CellNet.from_config(sample_config)
        lightning_model = PTM2CellNetLightning(model, sample_config)

        # 创建Trainer（使用CPU进行快速测试）
        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        # 训练
        trainer.fit(lightning_model, datamodule=datamodule)

        # 验证训练完成（max_epochs=2，训练完成后current_epoch应该>=1）
        assert trainer.current_epoch >= 1

    def test_model_checkpoint(self, sample_config, sample_data):
        """测试模型保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 准备数据
            preprocessor = DataPreprocessor(sample_config)
            train_df, val_df, test_df = preprocessor.preprocess_pipeline(sample_data)

            datamodule = PTMDataModule(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                config=sample_config,
            )

            # 创建模型
            model = PTM2CellNet.from_config(sample_config)
            lightning_model = PTM2CellNetLightning(model, sample_config)

            # 创建带检查点的Trainer
            from lightning.pytorch.callbacks import ModelCheckpoint

            checkpoint_callback = ModelCheckpoint(
                dirpath=tmpdir,
                filename="test-model",
                save_top_k=1,
            )

            trainer = L.Trainer(
                accelerator="cpu",
                max_epochs=1,
                callbacks=[checkpoint_callback],
                logger=False,
                enable_progress_bar=False,
            )

            # 训练
            trainer.fit(lightning_model, datamodule=datamodule)

            # 验证检查点文件存在
            checkpoint_files = list(Path(tmpdir).glob("*.ckpt"))
            assert len(checkpoint_files) > 0

    def test_different_encoders(self, sample_config, sample_data):
        """测试不同编码器类型"""
        encoder_types = ["cnn", "transformer", "lstm"]

        for encoder_type in encoder_types:
            config = sample_config.copy()
            config["model"]["encoder_type"] = encoder_type

            # 创建模型
            model = PTM2CellNet.from_config(config)
            lightning_model = PTM2CellNetLightning(model, config)

            # 验证模型创建成功
            assert lightning_model is not None
            assert model.encoder_type == encoder_type


class TestConfigIntegration:
    """配置集成测试"""

    def test_config_loading(self):
        """测试配置文件加载"""
        from src.utils.config import Config

        # 测试默认配置加载
        config = Config.from_yaml("configs/lightning.yaml")

        assert config.get("model.encoder_type") is not None
        assert config.get("training.max_epochs") is not None
        assert config.get("data.max_sequence_length") is not None

    def test_pretrained_config_loading(self):
        """测试预训练模型配置加载"""
        from src.utils.config import Config

        config = Config.from_yaml("configs/pretrained/esm2_150m.yaml")

        assert config.get("model.encoder_type") == "esm2_150M"
        assert config.get("model.freeze_encoder") is not None

    def test_config_override(self):
        """测试配置覆盖"""
        from src.utils.config import Config

        config = Config.from_yaml("configs/lightning.yaml")

        original_epochs = config.get("training.max_epochs")
        config.set("training.max_epochs", 200)

        assert config.get("training.max_epochs") == 200
        assert config.get("training.max_epochs") != original_epochs


# 需要导入Path
from pathlib import Path
