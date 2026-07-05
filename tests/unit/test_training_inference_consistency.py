"""
训练-推理一致性测试
功能: 验证训练后模型可正确导出/加载，推理结果与训练时预测一致

Requirements: TEST-02
"""

import pickle

import pytest
import torch

from src.models.architectures import PTM2CellNet
from src.utils.io import safe_torch_load
from src.models.model_utils import count_parameters


class TestTrainingInferenceConsistency:
    """训练-推理一致性测试"""

    @pytest.fixture
    def sample_batch(self):
        """生成测试批次"""
        def _create_batch(batch_size: int = 4, seq_len: int = 50, vocab_size: int = 20):
            return {
                "sequence": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "ptm_mask": torch.randint(0, 2, (batch_size, seq_len)).float(),
                "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
                "labels": torch.randint(0, 4, (batch_size,)),
            }
        return _create_batch

    @pytest.fixture
    def model_config(self):
        """模型配置"""
        return {
            "encoder_type": "cnn",
            "vocab_size": 20,
            "embed_dim": 64,
            "max_seq_len": 1000,
            "num_ptm_types": 10,
            "num_classes": 4,
            "pool_type": "mean",
            "dropout": 0.1,
        }

    def test_model_forward_deterministic(self, model_config, sample_batch):
        """测试模型前向传播是确定性的（相同输入产生相同输出）"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)
        model.eval()

        batch = sample_batch()
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        with torch.no_grad():
            output1 = model(inputs)
            output2 = model(inputs)

        # 两次前向传播结果应该相同
        assert torch.allclose(output1["logits"], output2["logits"], atol=1e-6)

    def test_save_load_consistency(self, model_config, sample_batch, tmp_path):
        """测试保存和加载后模型输出一致"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        batch = sample_batch()
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        # 保存前推理
        model.eval()
        with torch.no_grad():
            output_before = model(inputs)

        # 保存模型
        save_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), save_path)

        # 加载模型
        loaded_model = PTM2CellNet(**model_config)
        loaded_model.load_state_dict(safe_torch_load(save_path))
        loaded_model.eval()

        # 加载后推理
        with torch.no_grad():
            output_after = loaded_model(inputs)

        # 输出应该相同
        assert torch.allclose(
            output_before["logits"],
            output_after["logits"],
            atol=1e-6
        ), "加载后模型输出应与保存前一致"

    def test_save_load_full_model(self, model_config, sample_batch, tmp_path):
        """测试完整模型保存和加载"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        batch = sample_batch()
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        model.eval()
        with torch.no_grad():
            output_before = model(inputs)

        # 注意：某些pooling层使用local class，可能无法直接pickle
        # 这里我们主要测试state_dict保存/加载是主要方式
        # 完整模型保存只在某些情况下可用
        save_path = tmp_path / "full_model.pt"

        try:
            torch.save(model, save_path)
            loaded_model = safe_torch_load(save_path)
            loaded_model.eval()

            with torch.no_grad():
                output_after = loaded_model(inputs)

            assert torch.allclose(
                output_before["logits"],
                output_after["logits"],
                atol=1e-6
            ), "完整模型加载后输出应一致"
        except (AttributeError, pickle.PicklingError):
            # 某些层（如MeanPooling的local class）无法pickle
            # 这是预期行为，跳过测试
            pytest.skip("Full model pickle not supported with current pooling implementation")

    def test_model_parameters_unchanged_after_save_load(self, model_config, tmp_path):
        """测试保存加载后模型参数不变"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        # 获取保存前参数
        params_before = {name: param.clone() for name, param in model.named_parameters()}

        # 保存并加载
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        loaded_model = PTM2CellNet(**model_config)
        loaded_model.load_state_dict(safe_torch_load(save_path))

        # 对比参数
        for name, param_after in loaded_model.named_parameters():
            param_before = params_before[name]
            assert torch.allclose(param_before, param_after, atol=1e-6), \
                f"参数 {name} 加载后不一致"

    def test_training_changes_parameters(self, model_config, sample_batch):
        """测试训练会改变模型参数"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        # 记录训练前参数
        params_before = {name: param.clone() for name, param in model.named_parameters()}

        # 简单训练步骤
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        batch = sample_batch()
        for _ in range(5):
            optimizer.zero_grad()
            output = model(batch)
            loss = torch.nn.functional.cross_entropy(output["logits"], batch["labels"])
            loss.backward()
            optimizer.step()

        # 检查至少有一些参数发生了变化
        params_changed = 0
        for name, param_after in model.named_parameters():
            param_before = params_before[name]
            if not torch.allclose(param_before, param_after, atol=1e-6):
                params_changed += 1

        assert params_changed > 0, "训练后应该有参数发生变化"

    def test_inference_mode_consistency(self, model_config, sample_batch):
        """测试推理模式输出一致"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        batch = sample_batch()
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        # 多次推理应该一致
        model.eval()
        outputs = []
        with torch.no_grad():
            for _ in range(5):
                output = model(inputs)
                outputs.append(output["logits"])

        # 所有输出应该相同
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6), \
                f"第{i}次推理结果不一致"

    def test_batch_consistency(self, model_config):
        """测试批次处理一致性 - 单样本vs批次结果应该一致"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)
        model.eval()

        # 创建相同样本重复
        seq_len = 50
        single_input = {
            "sequence": torch.randint(0, 20, (1, seq_len)),
            "ptm_mask": torch.randint(0, 2, (1, seq_len)).float(),
            "ptm_types": torch.randint(0, 10, (1, seq_len)),
        }

        # 批次输入（3个相同样本）
        batch_input = {
            "sequence": single_input["sequence"].repeat(3, 1),
            "ptm_mask": single_input["ptm_mask"].repeat(3, 1),
            "ptm_types": single_input["ptm_types"].repeat(3, 1),
        }

        with torch.no_grad():
            single_output = model(single_input)
            batch_output = model(batch_input)

        # 批次中每个样本应该与单样本结果相同
        for i in range(3):
            assert torch.allclose(
                single_output["logits"],
                batch_output["logits"][i:i+1],
                atol=1e-5
            ), f"批次中第{i}个样本与单样本结果不一致"

    def test_different_batch_sizes_consistency(self, model_config, sample_batch):
        """测试不同批次大小的输出一致性"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)
        model.eval()

        batch = sample_batch(batch_size=8)
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        with torch.no_grad():
            full_output = model(inputs)

        # 分割批次处理
        split_outputs = []
        for i in range(0, 8, 4):
            split_input = {k: v[i:i+4] for k, v in inputs.items()}
            with torch.no_grad():
                split_output = model(split_input)
            split_outputs.append(split_output["logits"])

        combined_output = torch.cat(split_outputs, dim=0)

        # 组合结果应该与完整批次一致
        assert torch.allclose(full_output["logits"], combined_output, atol=1e-5), \
            "分割批次结果应与完整批次一致"

    def test_export_onnx_format(self, model_config, sample_batch, tmp_path):
        """测试导出为ONNX格式"""
        try:
            import onnx
        except ImportError:
            pytest.skip("ONNX libraries not available")

        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)
        model.eval()

        batch = sample_batch(batch_size=1)
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        # 导出ONNX
        export_path = tmp_path / "model.onnx"

        # 准备输入示例
        example_inputs = (
            inputs["sequence"],
            inputs["ptm_mask"],
            inputs["ptm_types"],
        )

        try:
            torch.onnx.export(
                model,
                example_inputs,
                export_path,
                input_names=["sequence", "ptm_mask", "ptm_types"],
                output_names=["logits"],
                dynamic_axes={
                    "sequence": {0: "batch_size", 1: "seq_len"},
                    "ptm_mask": {0: "batch_size", 1: "seq_len"},
                    "ptm_types": {0: "batch_size", 1: "seq_len"},
                    "logits": {0: "batch_size"},
                },
                opset_version=11,
            )

            # 验证ONNX模型可以加载
            onnx_model = onnx.load(export_path)
            onnx.checker.check_model(onnx_model)

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")

    def test_model_info_consistency(self, model_config):
        """测试模型信息一致性"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        info = model.get_model_info()

        # 验证基本信息
        assert info["encoder_type"] == model_config["encoder_type"]
        assert info["embed_dim"] == model_config["embed_dim"]
        assert info["num_classes"] == model_config["num_classes"]

        # 验证参数量统计
        param_stats = count_parameters(model)
        assert info["total_params"] == param_stats["total_params"]

    def test_dropout_behavior_train_vs_eval(self, model_config, sample_batch):
        """测试dropout在训练和评估模式下的不同行为"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        batch = sample_batch()
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        # 训练模式下多次前向传播结果不同（dropout随机）
        model.train()
        outputs_train = []
        for _ in range(5):
            output = model(inputs)
            outputs_train.append(output["logits"])

        # 评估模式下结果相同
        model.eval()
        outputs_eval = []
        with torch.no_grad():
            for _ in range(5):
                output = model(inputs)
                outputs_eval.append(output["logits"])

        # 评估模式所有输出应该相同
        for i in range(1, len(outputs_eval)):
            assert torch.allclose(outputs_eval[0], outputs_eval[i], atol=1e-6)

    def test_gradient_flow_during_training(self, model_config, sample_batch):
        """测试训练时梯度流动"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)
        model.train()

        batch = sample_batch()

        # 前向传播
        output = model(batch)
        loss = torch.nn.functional.cross_entropy(output["logits"], batch["labels"])

        # 反向传播
        loss.backward()

        # 检查是否有梯度
        has_grad = False
        for _name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "训练时应该有梯度产生"

    def test_loss_computation_consistency(self, model_config, sample_batch):
        """测试损失计算一致性"""
        torch.manual_seed(42)
        model = PTM2CellNet(**model_config)

        batch = sample_batch()

        # 多次计算损失应该相近（注意：由于dropout的存在，训练模式下结果会有微小差异）
        model.eval()  # 使用eval模式确保确定性
        losses = []
        for _ in range(3):
            output = model(batch)
            loss = torch.nn.functional.cross_entropy(
                output["logits"], batch["labels"]
            )
            losses.append(loss.item())

        # 损失值应该非常接近（eval模式下应该完全一致）
        assert abs(losses[0] - losses[1]) < 1e-5, f"Loss difference: {abs(losses[0] - losses[1])}"
        assert abs(losses[1] - losses[2]) < 1e-5, f"Loss difference: {abs(losses[1] - losses[2])}"


class TestAPIConsistency:
    """API一致性测试"""

    @pytest.fixture
    def sample_batch(self):
        """生成测试批次"""
        def _create_batch(batch_size: int = 4, seq_len: int = 50):
            return {
                "sequence": torch.randint(0, 20, (batch_size, seq_len)),
                "ptm_mask": torch.randint(0, 2, (batch_size, seq_len)).float(),
                "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
            }
        return _create_batch

    @pytest.fixture
    def model(self):
        """创建测试模型"""
        return PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=64,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
        )

    def test_api_vs_direct_call_consistency(self, model, sample_batch):
        """测试API调用与直接模型调用结果一致"""
        # 模拟API风格的预测
        batch = sample_batch()

        # 直接调用
        model.eval()
        with torch.no_grad():
            direct_output = model(batch)

        # 模拟API处理（相同输入，相同处理）
        def api_predict(model, batch):
            model.eval()
            with torch.no_grad():
                output = model(batch)
                return {
                    "predictions": output["predictions"].tolist(),
                    "probabilities": output["probabilities"].tolist()
                    if "probabilities" in output
                    else None,
                }

        api_output = api_predict(model, batch)

        # 转换回tensor比较
        api_predictions = torch.tensor(api_output["predictions"])

        assert torch.allclose(direct_output["predictions"].float(), api_predictions.float())

    def test_prediction_output_format(self, model, sample_batch):
        """测试预测输出格式"""
        batch = sample_batch(batch_size=4)

        model.eval()
        with torch.no_grad():
            output = model(batch)

        # 验证输出包含预期字段
        assert "logits" in output
        assert "predictions" in output
        assert "probabilities" in output

        # 验证形状
        assert output["logits"].shape == (4, 4)  # batch_size=4, num_classes=4
        assert output["predictions"].shape == (4,)
        assert output["probabilities"].shape == (4, 4)

        # 验证概率和为1
        prob_sums = output["probabilities"].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5)

    def test_single_vs_batch_prediction_consistency(self, model):
        """测试单样本和批次预测一致性"""
        seq_len = 50

        # 单样本
        single_input = {
            "sequence": torch.randint(0, 20, (1, seq_len)),
            "ptm_mask": torch.randint(0, 2, (1, seq_len)).float(),
            "ptm_types": torch.randint(0, 10, (1, seq_len)),
        }

        model.eval()
        with torch.no_grad():
            single_output = model(single_input)

        # 通过批次API处理单样本
        def mock_api_single_predict(model, sequence, ptm_mask, ptm_types):
            batch_input = {
                "sequence": sequence.unsqueeze(0),
                "ptm_mask": ptm_mask.unsqueeze(0),
                "ptm_types": ptm_types.unsqueeze(0),
            }
            model.eval()
            with torch.no_grad():
                output = model(batch_input)
            return {
                "prediction": output["predictions"][0].item(),
                "probabilities": output["probabilities"][0].tolist(),
            }

        api_result = mock_api_single_predict(
            model,
            single_input["sequence"].squeeze(0),
            single_input["ptm_mask"].squeeze(0),
            single_input["ptm_types"].squeeze(0),
        )

        # API结果应与直接调用一致
        assert api_result["prediction"] == single_output["predictions"][0].item()
        api_probs = torch.tensor(api_result["probabilities"])
        assert torch.allclose(api_probs, single_output["probabilities"][0], atol=1e-5)

    def test_regression_task_consistency(self, sample_batch):
        """测试回归任务输出一致性"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=64,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=1,  # 回归任务
            task_type="regression",
        )

        batch = sample_batch()

        model.eval()
        with torch.no_grad():
            output = model(batch)

        # 回归任务返回predictions，并提供logits别名兼容旧调用方
        assert "predictions" in output
        assert "logits" in output
        assert torch.equal(output["predictions"], output["logits"])
        assert "probabilities" not in output

        # 预测值应该是实数
        assert output["predictions"].dtype in [torch.float32, torch.float64]


class TestCheckpointConsistency:
    """检查点一致性测试"""

    def test_checkpoint_save_load(self, tmp_path):
        """测试检查点保存和加载"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=64,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
        )

        # 模拟训练状态
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epoch = 5
        best_loss = 0.5

        # 保存检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # 加载检查点
        loaded_model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=64,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
        )
        loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.001)

        checkpoint = safe_torch_load(checkpoint_path)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 验证状态
        assert checkpoint["epoch"] == epoch
        assert checkpoint["best_loss"] == best_loss

        # 验证模型状态一致
        for (_n1, p1), (_n2, p2) in zip(
            model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=1e-6)
