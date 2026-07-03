"""
checkpoint_utils 单元测试

覆盖统一 checkpoint 合约（审计报告 P0-3）：
    - :func:`extract_model_state_dict` 对裸 state_dict / Lightning .ckpt /
      旧格式 / 空字典 / 无法识别前缀的处理。
    - :func:`load_model` 消费 Lightning .ckpt、strict 模式、save/load 回环。

复用 ``tests/e2e/test_train_predict_cli.py::TestUnifiedCheckpointContract`` 中
已验证的 Lightning 前缀剥离模式。
"""

import pytest
import torch

from src.utils.checkpoint_utils import extract_model_state_dict
from src.utils.io import load_model


def _make_sd():
    """返回一个最小、确定性的裸 state_dict。"""
    return {
        "encoder.weight": torch.zeros(2, 2),
        "head.bias": torch.zeros(2),
    }


# ---------------------------------------------------------------------------
# extract_model_state_dict
# ---------------------------------------------------------------------------

class TestExtractModelStateDict:
    def test_extract_model_state_dict_lightning_prefix(self):
        """Lightning .ckpt：剥离 ``model.`` 前缀后返回裸权重。"""
        sd = _make_sd()
        lightning_ckpt = {
            "epoch": 1,
            "global_step": 5,
            "pytorch-lightning_version": "2.0",
            "state_dict": {f"model.{k}": v for k, v in sd.items()},
        }
        out = extract_model_state_dict(lightning_ckpt)
        assert set(out.keys()) == set(sd.keys())
        for k in sd:
            assert torch.equal(out[k], sd[k]), f"权重不一致: {k}"

    def test_extract_model_state_dict_bare_model(self):
        """裸 state_dict（值全为 Tensor）原样返回（同一对象）。"""
        sd = _make_sd()
        out = extract_model_state_dict(sd)
        assert out is sd
        assert set(out.keys()) == set(sd.keys())

    def test_extract_model_state_dict_legacy_format(self):
        """旧格式：``model_state_dict`` 键直接返回其中的权重。"""
        sd = _make_sd()
        ckpt = {"epoch": 0, "best_score": 0.9, "model_state_dict": sd}
        out = extract_model_state_dict(ckpt)
        assert out is sd
        assert set(out.keys()) == set(sd.keys())

    def test_extract_model_state_dict_empty(self):
        """空 dict 视为裸 state_dict，原样返回。"""
        empty: dict = {}
        out = extract_model_state_dict(empty)
        assert out is empty
        assert out == {}

    def test_extract_model_state_dict_wrong_prefix(self):
        """顶层含 ``state_dict`` 但键不带 ``model.`` 前缀：保持原样返回。

        ``_strip_lightning_prefix`` 仅在 **所有** 键都以 ``model.`` 开头时才剥离，
        因此此处不应剥离，返回的 dict 键名应保留原前缀。
        """
        sd = {"layer.weight": torch.zeros(2, 2), "layer.bias": torch.zeros(2)}
        ckpt = {"epoch": 1, "state_dict": sd}
        out = extract_model_state_dict(ckpt)
        # 未剥离前缀——键名与原始一致
        assert set(out.keys()) == set(sd.keys())
        assert out is sd

    def test_extract_model_state_dict_non_dict_raises(self):
        """非 dict 输入抛 TypeError。"""
        with pytest.raises(TypeError):
            extract_model_state_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_extract_model_state_dict_unrecognized_raises(self):
        """无法识别结构的 dict 抛 TypeError。"""
        with pytest.raises(TypeError):
            extract_model_state_dict({"epoch": 1, "optimizer": "adam"})


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def _make_model(self):
        from src.models.architectures import PTM2CellNet

        return PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=16,
            max_seq_len=64,
            num_ptm_types=5,
            num_classes=4,
        )

    def test_load_model_lightning_ckpt(self, tmp_path):
        """load_model 可直接消费 Lightning .ckpt 并加载到裸模型。"""
        model = self._make_model()
        sd = model.state_dict()
        lightning_ckpt = {
            "epoch": 1,
            "global_step": 5,
            "pytorch-lightning_version": "2.0",
            "state_dict": {f"model.{k}": v for k, v in sd.items()},
        }
        ckpt = tmp_path / "lightning.ckpt"
        torch.save(lightning_ckpt, ckpt)

        fresh = self._make_model()
        load_model(fresh, str(ckpt))  # strict=True 默认
        for k in sd:
            assert torch.allclose(sd[k], fresh.state_dict()[k]), f"权重不一致: {k}"

    def test_load_model_strict_mode(self, tmp_path):
        """strict=True：键不匹配时抛 RuntimeError；匹配时不抛错。"""
        model = self._make_model()
        sd = model.state_dict()

        # 1) 匹配：strict=True 不抛错
        ckpt_ok = tmp_path / "matched.ckpt"
        torch.save(sd, ckpt_ok)
        fresh = self._make_model()
        load_model(fresh, str(ckpt_ok), strict=True)
        for k in sd:
            assert torch.allclose(sd[k], fresh.state_dict()[k]), f"权重不一致: {k}"

        # 2) 不匹配（裁掉一个键）：strict=True 抛 RuntimeError
        truncated = {k: v for i, (k, v) in enumerate(sd.items()) if i == 0}
        ckpt_bad = tmp_path / "truncated.ckpt"
        torch.save(truncated, ckpt_bad)
        fresh2 = self._make_model()
        with pytest.raises(RuntimeError):
            load_model(fresh2, str(ckpt_bad), strict=True)

        # 3) strict=False：不匹配时不抛错（仅记录）
        fresh3 = self._make_model()
        load_model(fresh3, str(ckpt_bad), strict=False)

    def test_load_model_file_not_found(self, tmp_path):
        """权重文件不存在抛 FileNotFoundError。"""
        fresh = self._make_model()
        with pytest.raises(FileNotFoundError):
            load_model(fresh, str(tmp_path / "nonexistent.ckpt"))

    def test_save_and_load_roundtrip(self, tmp_path):
        """save -> load 回环：权重逐键一致。"""
        model = self._make_model()
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}

        ckpt = tmp_path / "roundtrip.pt"
        torch.save(model.state_dict(), ckpt)

        fresh = self._make_model()
        load_model(fresh, str(ckpt), strict=True)
        sd_after = fresh.state_dict()
        for k in sd_before:
            assert torch.allclose(sd_before[k], sd_after[k]), f"回环权重不一致: {k}"
