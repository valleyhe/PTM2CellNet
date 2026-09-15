"""旧 DAVF checkpoint 兼容层测试。

覆盖：
1. 旧架构 checkpoint（delta_mlp 参数块）经迁移后可由 DAVFInferenceModule 加载，
   model_source == "davf"（而非 zero_fallback）；
2. 加载后特征非零、随方向变化（方向敏感性）；
3. 全掩码输入无 NaN（鲁棒性）。
"""

import torch

from src.models.davf_inference import DAVFInferenceConfig, DAVFInferenceModule
from src.models.ptm_direction_mapper import PTMDirectionMapperOutput


def _make_module(checkpoint_path: str) -> DAVFInferenceModule:
    cfg = DAVFInferenceConfig(
        checkpoint_path=checkpoint_path,
        state_space="scvi_latent",
        feature_dim=128,
        hidden_dim=256,
        latent_dim=64,
        num_genes=5000,
        num_steps=50,
        freeze=True,
        device="cpu",
    )
    return DAVFInferenceModule(cfg)


def test_legacy_checkpoint_loads_as_davf():
    mod = _make_module("checkpoints/latent_davf_ibd_norman/best_model.safe.pt")
    assert mod.model_source == "davf"
    assert type(mod.latent_davf).__name__ == "LegacyLatentDAVF"


def test_legacy_features_are_direction_sensitive():
    mod = _make_module("checkpoints/latent_davf_ibd_norman/best_model.safe.pt")

    def feats(g: int, d: int) -> torch.Tensor:
        out = PTMDirectionMapperOutput(
            gene_ids=torch.tensor([[g]]),
            directions=torch.tensor([[d]]),
            attention_mask=torch.ones(1, 1),
        )
        return mod(out).davf_features.detach()

    f_ko = feats(100, 0)
    f_oe = feats(100, 2)
    assert torch.isfinite(f_ko).all()
    assert f_ko.std() > 0.05  # 非退化特征
    # 方向不同 → 特征余弦距离显著（< 0.99 即非恒等）
    cos = torch.nn.functional.cosine_similarity(f_ko, f_oe, dim=-1).item()
    assert cos < 0.99


def test_legacy_all_masked_no_nan():
    mod = _make_module("checkpoints/latent_davf_ibd_norman/best_model.safe.pt")
    out = PTMDirectionMapperOutput(
        gene_ids=torch.zeros(1, 1, dtype=torch.long),
        directions=torch.zeros(1, 1, dtype=torch.long),
        attention_mask=torch.zeros(1, 1),
    )
    feats = mod(out).davf_features.detach()
    assert torch.isfinite(feats).all()
