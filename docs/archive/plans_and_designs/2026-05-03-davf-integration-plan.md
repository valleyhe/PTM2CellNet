# DAVF集成实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将DAVF方向感知模型集成到PTM2CellNet中，实现PTM修饰→信号通路效应预测的级联融合架构。

**Architecture:** 从source_code.zip提取DAVF核心模块(davf.py, latent_davf.py, biperturb.py等)放入项目。新增PTMDirectionMapper将PTM修饰映射为DAVF输入格式，新增DAVFInferenceModule封装推理逻辑，新增DeltaProjection压缩DAVF输出。修改PTM2CellNet架构使其在forward中并行执行DAVF推理并拼接特征到predictor。

**Tech Stack:** PyTorch, scvi-tools, anndata, transformers(Geneformer)

---

## Task 1: 解压并导入DAVF核心源文件

**Files:**
- Create: `src/models/davf.py`
- Create: `src/models/latent_davf.py`
- Create: `src/models/biperturb.py`
- Create: `src/models/davf_attention.py`
- Create: `src/models/geneformer_embedding.py`
- Create: `src/models/delta_predictor.py`
- Create: `src/models/scvi_adapter.py`
- Create: `src/models/davf_checkpoint_utils.py`

**Step 1: 从source_code.zip提取DAVF核心文件**

```bash
cd /home/scu/PTM2CellNet
# 提取核心模型文件到临时目录
unzip -o source_code.zip \
  src/models/davf.py \
  src/models/latent_davf.py \
  src/models/biperturb.py \
  src/models/attention.py \
  src/models/geneformer_embedding.py \
  src/models/delta_predictor.py \
  src/models/scvi_adapter.py \
  src/models/checkpoint_utils.py \
  -d /tmp/davf_extract
```

**Step 2: 复制文件到项目，attention.py重命名避免冲突**

```bash
cp /tmp/davf_extract/src/models/davf.py src/models/davf.py
cp /tmp/davf_extract/src/models/latent_davf.py src/models/latent_davf.py
cp /tmp/davf_extract/src/models/biperturb.py src/models/biperturb.py
cp /tmp/davf_extract/src/models/attention.py src/models/davf_attention.py
cp /tmp/davf_extract/src/models/geneformer_embedding.py src/models/geneformer_embedding.py
cp /tmp/davf_extract/src/models/delta_predictor.py src/models/delta_predictor.py
cp /tmp/davf_extract/src/models/scvi_adapter.py src/models/scvi_adapter.py
cp /tmp/davf_extract/src/models/checkpoint_utils.py src/models/davf_checkpoint_utils.py
```

**Step 3: 修正import路径**

所有提取的文件中 `from src.models.attention import` 改为 `from src.models.davf_attention import`。
确保文件中的相对导入路径与项目结构一致。

**Step 4: 验证文件可导入**

```bash
cd /home/scu/PTM2CellNet
python -c "from src.models.davf_attention import DirectionAwareAttention, MultiTargetAttention; print('OK')"
python -c "from src.models.biperturb import BiPerturbConfig, DirectionEncoder, BiPerturbEncoder; print('OK')"
python -c "from src.models.davf import DAVFConfig, TimeEncoder; print('OK')"
python -c "from src.models.latent_davf import LatentDAVFConfig, LatentDAVF; print('OK')"
```

**Step 5: Commit**

```bash
git add src/models/davf.py src/models/latent_davf.py src/models/biperturb.py \
  src/models/davf_attention.py src/models/geneformer_embedding.py \
  src/models/delta_predictor.py src/models/scvi_adapter.py \
  src/models/davf_checkpoint_utils.py
git commit -m "feat: import DAVF core modules from source_code.zip"
```

---

## Task 2: 解压预训练模型检查点

**Files:**
- Create: `checkpoints/` directory tree

**Step 1: 解压model_checkpoints.zip**

```bash
cd /home/scu/PTM2CellNet
unzip -o model_checkpoints.zip -d .
```

此操作会创建:
- `checkpoints/davf/model_a_4018/best_model.pt`
- `checkpoints/davf/model_a_finetuned/best_model.pt`
- `checkpoints/latent_davf_ibd_norman/best_model.pt`
- `checkpoints/latent_davf_ibd_replogle/best_model.pt`
- `checkpoints/latent_davf_ibd_tcm_aware/best_model.pt`
- `checkpoints/scvi/ibd_norman_model/model.pt`
- `checkpoints/geneformer_embeddings.pt`
- 等

**Step 2: 确认检查点文件存在**

```bash
ls -la checkpoints/latent_davf_ibd_norman/best_model.pt
ls -la checkpoints/geneformer_embeddings.pt
ls -la checkpoints/scvi/ibd_norman_model/model.pt
```

**Step 3: 添加checkpoints到.gitignore**

```bash
echo "checkpoints/" >> .gitignore
```

**Step 4: Commit .gitignore更新**

```bash
git add .gitignore
git commit -m "chore: add checkpoints/ to .gitignore"
```

---

## Task 3: 创建PTMDirectionMapper

**Files:**
- Create: `src/models/ptm_direction_mapper.py`
- Test: `tests/unit/test_ptm_direction_mapper.py`

**Step 1: 编写测试**

```python
# tests/unit/test_ptm_direction_mapper.py
"""Tests for PTMDirectionMapper."""
import pytest
import torch
from src.models.ptm_direction_mapper import PTMDirectionMapper


class TestPTMDirectionMapper:
    def setup_method(self):
        self.mapper = PTMDirectionMapper(num_genes=5000)

    def test_phosphorylation_maps_to_oe(self):
        """磷酸化默认映射为OE(activate)方向"""
        result = self.mapper.map_ptm_to_perturbation(
            ptm_types=["Phosphorylation"],
            gene_names=["AKT1"],
        )
        assert result["directions"][0, 0].item() == 2  # OE

    def test_ubiquitination_maps_to_ko(self):
        """泛素化映射为KO(inhibit)方向"""
        result = self.mapper.map_ptm_to_perturbation(
            ptm_types=["Ubiquitination"],
            gene_names=["TP53"],
        )
        assert result["directions"][0, 0].item() == 0  # KO

    def test_batch_mapping(self):
        """批量PTM映射"""
        result = self.mapper.map_ptm_to_perturbation(
            ptm_types=["Phosphorylation", "Ubiquitination", "Acetylation"],
            gene_names=["MAPK1", "NFKBIA", "STAT3"],
        )
        assert result["gene_ids"].shape == (1, 3)
        assert result["directions"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)

    def test_unknown_gene_gets_masked(self):
        """未知基因应被mask掉"""
        result = self.mapper.map_ptm_to_perturbation(
            ptm_types=["Phosphorylation"],
            gene_names=["UNKNOWN_GENE_XYZ"],
        )
        assert result["attention_mask"][0, 0].item() == 0

    def test_output_tensors_correct_dtype(self):
        """输出张量类型正确"""
        result = self.mapper.map_ptm_to_perturbation(
            ptm_types=["Phosphorylation"],
            gene_names=["AKT1"],
        )
        assert result["gene_ids"].dtype == torch.long
        assert result["directions"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.float
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_ptm_direction_mapper.py -v
```

Expected: FAIL (module not found)

**Step 3: 实现PTMDirectionMapper**

```python
# src/models/ptm_direction_mapper.py
"""PTM Direction Mapper: 将PTM修饰信息映射为DAVF扰动输入格式。

映射规则:
- Phosphorylation → OE (activate, direction=2)
- Acetylation → OE (activate, direction=2)
- Ubiquitination → KO (inhibit, direction=0)
- Sumoylation → KO (inhibit, direction=0)
- Succinylation → KO (inhibit, direction=0)
- Methylation → context-dependent (default KD, direction=1)
"""

import torch
from typing import Dict, List, Optional


# PTM类型到默认方向的映射
PTM_DIRECTION_MAP = {
    "Phosphorylation": 2,   # OE (activate)
    "phosphorylation": 2,
    "Acetylation": 2,       # OE (activate)
    "acetylation": 2,
    "Methylation": 1,       # KD (context-dependent, default neutral)
    "methylation": 1,
    "Ubiquitination": 0,    # KO (inhibit/degrade)
    "ubiquitination": 0,
    "Sumoylation": 0,       # KO (inhibit)
    "sumoylation": 0,
    "Succinylation": 0,     # KO (inhibit)
    "succinylation": 0,
}


class PTMDirectionMapper:
    """将PTM修饰事件映射为DAVF模型的输入格式(gene_ids, directions, attention_mask)。"""

    def __init__(
        self,
        num_genes: int = 5000,
        gene_name_list: Optional[List[str]] = None,
        custom_direction_map: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            num_genes: DAVF模型的基因数量
            gene_name_list: 基因名到索引的有序列表(来自DAVF checkpoint)
            custom_direction_map: 自定义PTM→direction映射，覆盖默认值
        """
        self.num_genes = num_genes
        self.gene_name_list = gene_name_list or []
        self.gene_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.gene_name_list)
        }
        self.direction_map = {**PTM_DIRECTION_MAP}
        if custom_direction_map:
            self.direction_map.update(custom_direction_map)

    def set_gene_names(self, gene_names: List[str]) -> None:
        """设置基因名列表(从checkpoint加载后调用)。"""
        self.gene_name_list = gene_names
        self.gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    def _get_direction(self, ptm_type: str) -> int:
        """获取PTM类型对应的方向编码。"""
        return self.direction_map.get(ptm_type, 1)  # 默认KD(neutral)

    def _get_gene_index(self, gene_name: str) -> int:
        """获取基因名对应的索引，未找到返回-1。"""
        return self.gene_to_idx.get(gene_name, -1)

    def map_ptm_to_perturbation(
        self,
        ptm_types: List[str],
        gene_names: List[str],
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """将PTM修饰列表映射为DAVF输入张量。

        Args:
            ptm_types: PTM修饰类型列表 (如 ["Phosphorylation", "Ubiquitination"])
            gene_names: 对应的基因名列表 (如 ["AKT1", "TP53"])
            batch_size: 批次大小(默认1, 所有PTM作为同一样本的多目标)

        Returns:
            Dict with:
            - gene_ids: [batch_size, num_targets] long tensor
            - directions: [batch_size, num_targets] long tensor (0=KO, 1=KD, 2=OE)
            - attention_mask: [batch_size, num_targets] float tensor (1=valid, 0=masked)
        """
        if len(ptm_types) != len(gene_names):
            raise ValueError(
                f"ptm_types and gene_names must have same length, "
                f"got {len(ptm_types)} vs {len(gene_names)}"
            )

        num_targets = len(ptm_types)
        gene_ids = torch.zeros(batch_size, num_targets, dtype=torch.long)
        directions = torch.zeros(batch_size, num_targets, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, num_targets, dtype=torch.float)

        for i, (ptm_type, gene_name) in enumerate(zip(ptm_types, gene_names)):
            gene_idx = self._get_gene_index(gene_name)
            direction = self._get_direction(ptm_type)

            if gene_idx >= 0:
                gene_ids[0, i] = gene_idx
                directions[0, i] = direction
                attention_mask[0, i] = 1.0
            else:
                # 未知基因: 保持0索引但mask掉
                gene_ids[0, i] = 0
                directions[0, i] = direction
                attention_mask[0, i] = 0.0

        # 如果batch_size > 1, 复制到所有batch
        if batch_size > 1:
            gene_ids = gene_ids.expand(batch_size, -1).clone()
            directions = directions.expand(batch_size, -1).clone()
            attention_mask = attention_mask.expand(batch_size, -1).clone()

        return {
            "gene_ids": gene_ids,
            "directions": directions,
            "attention_mask": attention_mask,
        }
```

**Step 4: 运行测试**

```bash
pytest tests/unit/test_ptm_direction_mapper.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/ptm_direction_mapper.py tests/unit/test_ptm_direction_mapper.py
git commit -m "feat: add PTMDirectionMapper for PTM→DAVF input mapping"
```

---

## Task 4: 创建DAVFInferenceModule

**Files:**
- Create: `src/models/davf_inference.py`
- Test: `tests/unit/test_davf_inference.py`

**Step 1: 编写测试**

```python
# tests/unit/test_davf_inference.py
"""Tests for DAVFInferenceModule."""
import pytest
import torch
from unittest.mock import patch, MagicMock
from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig


class TestDAVFInferenceConfig:
    def test_default_config(self):
        config = DAVFInferenceConfig()
        assert config.state_space == "scvi_latent"
        assert config.feature_dim == 128
        assert config.freeze is True
        assert config.num_genes == 5000

    def test_gene_space_config(self):
        config = DAVFInferenceConfig(state_space="gene")
        assert config.state_space == "gene"


class TestDAVFInferenceModule:
    def test_delta_projection_output_shape(self):
        """DeltaProjection输出维度正确"""
        module = DAVFInferenceModule.__new__(DAVFInferenceModule)
        module.config = DAVFInferenceConfig(num_genes=100, feature_dim=64)
        module._build_delta_projection()
        
        dummy_delta = torch.randn(4, 100)
        output = module.delta_projection(dummy_delta)
        assert output.shape == (4, 64)

    def test_freeze_parameters(self):
        """冻结模式下参数不需要梯度"""
        module = DAVFInferenceModule.__new__(DAVFInferenceModule)
        module.config = DAVFInferenceConfig(num_genes=100, feature_dim=64, freeze=True)
        module._build_delta_projection()
        module.davf_model = torch.nn.Linear(10, 10)
        module._freeze_davf()
        
        for p in module.davf_model.parameters():
            assert p.requires_grad is False
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_davf_inference.py -v
```

**Step 3: 实现DAVFInferenceModule**

```python
# src/models/davf_inference.py
"""DAVF Inference Module: 封装DAVF/LatentDAVF推理逻辑。

职责:
- 加载预训练DAVF checkpoint
- 支持gene空间和scVI latent空间推理
- 提供delta_expression → 压缩特征的projection
- 管理冻结/解冻状态
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DAVFInferenceConfig:
    """DAVF推理模块配置。"""
    state_space: str = "scvi_latent"  # "gene" | "scvi_latent"
    checkpoint_path: str = "checkpoints/latent_davf_ibd_norman/best_model.pt"
    scvi_model_path: str = "checkpoints/scvi/ibd_norman_model/model.pt"
    geneformer_path: str = "checkpoints/geneformer_embeddings.pt"
    feature_dim: int = 128  # DeltaProjection输出维度
    freeze: bool = True
    num_genes: int = 5000
    num_inference_steps: int = 10  # ODE积分步数


class DAVFInferenceModule(nn.Module):
    """DAVF推理封装模块。

    将DAVF模型的推理结果(delta_expression)转换为固定维度特征向量,
    用于与PTM2CellNet的序列特征级联融合。
    """

    def __init__(
        self,
        config: DAVFInferenceConfig,
        device: str = "cpu",
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.davf_model: Optional[nn.Module] = None
        self.scvi_adapter = None
        self.gene_names: List[str] = []

        # Delta projection: num_genes → feature_dim
        self._build_delta_projection()

        # 加载模型
        self._load_models()

        # 冻结DAVF参数
        if config.freeze:
            self._freeze_davf()

    def _build_delta_projection(self) -> None:
        """构建delta_expression到特征向量的投影网络。"""
        num_genes = self.config.num_genes
        feature_dim = self.config.feature_dim

        self.delta_projection = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def _load_models(self) -> None:
        """加载DAVF模型和相关组件。"""
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(
                f"DAVF checkpoint not found: {checkpoint_path}. "
                f"Module will output zero features until checkpoint is loaded."
            )
            return

        logger.info(f"Loading DAVF from {checkpoint_path}...")
        ckpt = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(ckpt, dict):
            raise ValueError("Invalid DAVF checkpoint format")

        # 提取gene_names
        self.gene_names = ckpt.get("gene_names", [])
        if not self.gene_names:
            self.gene_names = [f"GENE_{i}" for i in range(self.config.num_genes)]

        # 更新num_genes
        actual_num_genes = len(self.gene_names)
        if actual_num_genes != self.config.num_genes:
            logger.info(
                f"Updating num_genes from {self.config.num_genes} to {actual_num_genes}"
            )
            self.config.num_genes = actual_num_genes
            # 重建projection
            self._build_delta_projection()

        # 根据state_space加载不同模型
        if self.config.state_space == "scvi_latent":
            self._load_latent_davf(ckpt)
        else:
            self._load_gene_davf(ckpt)

    def _load_latent_davf(self, ckpt: dict) -> None:
        """加载LatentDAVF模型。"""
        from src.models.latent_davf import LatentDAVF, LatentDAVFConfig
        from src.models.delta_predictor import DeltaPredictor

        ckpt_config = ckpt.get("config")
        if isinstance(ckpt_config, LatentDAVFConfig):
            config = ckpt_config
        elif isinstance(ckpt_config, dict):
            config = LatentDAVFConfig(**ckpt_config)
        else:
            config = LatentDAVFConfig(num_genes=self.config.num_genes)

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state_dict is None:
            raise ValueError("Checkpoint missing model_state_dict")

        # 检测模型架构
        state_keys = set(state_dict.keys())
        uses_delta_mlp = any("delta_mlp" in k for k in state_keys)

        # 提取预训练嵌入
        pretrained_emb = None
        if "gene_embed_table.weight" in state_dict:
            pretrained_emb = state_dict["gene_embed_table.weight"].clone()

        if uses_delta_mlp:
            self.davf_model = DeltaPredictor(
                config,
                gene_names=self.gene_names,
                pretrained_gene_embeddings=pretrained_emb,
            )
        else:
            self.davf_model = LatentDAVF(
                config,
                gene_names=self.gene_names,
                pretrained_gene_embeddings=pretrained_emb,
            )

        self.davf_model.load_state_dict(state_dict, strict=False)
        self.davf_model.to(self.device)
        self.davf_model.eval()
        logger.info(f"LatentDAVF loaded: {type(self.davf_model).__name__}")

    def _load_gene_davf(self, ckpt: dict) -> None:
        """加载Gene空间DAVF模型。"""
        from src.models.davf import DAVF, DAVFConfig

        ckpt_config = ckpt.get("config")
        if isinstance(ckpt_config, dict):
            ckpt_config.setdefault("condition_injection", "hybrid")
            ckpt_config.setdefault("modulation_dim", 128)
            ckpt_config.setdefault("num_kv_heads", 8)
            config = DAVFConfig(**ckpt_config)
        elif isinstance(ckpt_config, DAVFConfig):
            config = ckpt_config
        else:
            config = DAVFConfig(num_genes=self.config.num_genes)

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state_dict is None:
            raise ValueError("Checkpoint missing model_state_dict")

        self.davf_model = DAVF(config, gene_names=self.gene_names)
        self.davf_model.load_state_dict(state_dict, strict=False)
        self.davf_model.to(self.device)
        self.davf_model.eval()
        logger.info("Gene-space DAVF loaded")

    def _freeze_davf(self) -> None:
        """冻结DAVF模型参数。"""
        if self.davf_model is not None:
            for p in self.davf_model.parameters():
                p.requires_grad = False

    def unfreeze_davf(self, layers: Optional[List[str]] = None) -> None:
        """解冻DAVF模型(用于微调)。

        Args:
            layers: 要解冻的层名前缀列表。None表示全部解冻。
        """
        if self.davf_model is None:
            return
        if layers is None:
            for p in self.davf_model.parameters():
                p.requires_grad = True
        else:
            for name, p in self.davf_model.named_parameters():
                if any(name.startswith(prefix) for prefix in layers):
                    p.requires_grad = True

    def forward(
        self,
        gene_ids: torch.Tensor,
        directions: torch.Tensor,
        attention_mask: torch.Tensor,
        control_expression: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """推理并返回压缩的DAVF特征。

        Args:
            gene_ids: [B, K] 目标基因索引
            directions: [B, K] 方向编码 (0=KO, 1=KD, 2=OE)
            attention_mask: [B, K] 有效mask
            control_expression: [B, num_genes] 控制组表达(可选，默认零向量)

        Returns:
            davf_features: [B, feature_dim] 压缩后的DAVF特征
        """
        B = gene_ids.shape[0]
        device = gene_ids.device

        # 如果模型未加载, 返回零向量
        if self.davf_model is None:
            return torch.zeros(B, self.config.feature_dim, device=device)

        # 准备控制表达
        if control_expression is None:
            control_expression = torch.zeros(
                B, self.config.num_genes, device=device
            )

        # DAVF推理
        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            if self.config.state_space == "gene":
                # Gene空间: 直接调用DAVF.predict
                x_1_pred = self.davf_model.predict(
                    x_0=control_expression,
                    gene_ids=gene_ids,
                    directions=directions,
                    num_steps=self.config.num_inference_steps,
                    attention_mask=attention_mask,
                )
                delta_expression = x_1_pred - control_expression
            else:
                # Latent空间: 使用LatentDAVF/DeltaPredictor
                # DeltaPredictor直接预测latent delta
                from src.models.delta_predictor import DeltaPredictor
                if isinstance(self.davf_model, DeltaPredictor):
                    delta_z = self.davf_model.predict(
                        z_0=control_expression[:, :self.davf_model.config.latent_dim],
                        gene_ids=gene_ids,
                        directions=directions,
                        attention_mask=attention_mask,
                    )
                    # 将latent delta扩展到num_genes维度(简化处理)
                    # 实际使用时应通过scVI decode, 这里用线性扩展近似
                    latent_dim = delta_z.shape[-1]
                    if not hasattr(self, '_latent_expand'):
                        self._latent_expand = nn.Linear(
                            latent_dim, self.config.num_genes
                        ).to(device)
                    delta_expression = self._latent_expand(delta_z)
                else:
                    # LatentDAVF with ODE
                    latent_dim = self.davf_model.config.latent_dim
                    z_0 = control_expression[:, :latent_dim]
                    x_1_pred = self.davf_model.predict(
                        z_0=z_0,
                        gene_ids=gene_ids,
                        directions=directions,
                        num_steps=self.config.num_inference_steps,
                        attention_mask=attention_mask,
                    )
                    delta_z = x_1_pred - z_0
                    if not hasattr(self, '_latent_expand'):
                        self._latent_expand = nn.Linear(
                            latent_dim, self.config.num_genes
                        ).to(device)
                    delta_expression = self._latent_expand(delta_z)

        # 压缩为特征向量
        davf_features = self.delta_projection(delta_expression)
        return davf_features

    def get_gene_names(self) -> List[str]:
        """返回DAVF使用的基因名列表。"""
        return self.gene_names
```

**Step 4: 运行测试**

```bash
pytest tests/unit/test_davf_inference.py -v
```

**Step 5: Commit**

```bash
git add src/models/davf_inference.py tests/unit/test_davf_inference.py
git commit -m "feat: add DAVFInferenceModule for DAVF model inference wrapper"
```

---

## Task 5: 修改PTM2CellNet架构支持DAVF级联融合

**Files:**
- Modify: `src/models/architectures.py`
- Test: `tests/unit/test_architectures_davf.py`

**Step 1: 编写测试**

```python
# tests/unit/test_architectures_davf.py
"""Tests for PTM2CellNet with DAVF integration."""
import pytest
import torch
from src.models.architectures import PTM2CellNet


class TestPTM2CellNetWithDAVF:
    def test_davf_disabled_backward_compatible(self):
        """use_davf=False时行为不变"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            num_classes=4,
            use_davf=False,
        )
        batch = {
            "sequence": torch.randint(0, 21, (2, 100)),
            "ptm_types": torch.randint(0, 5, (2, 100)),
        }
        output = model(batch)
        assert "logits" in output
        assert output["logits"].shape == (2, 4)

    def test_davf_enabled_increases_predictor_input(self):
        """use_davf=True时predictor输入维度增加"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            num_classes=4,
            use_davf=True,
            davf_config={"feature_dim": 128, "num_genes": 100},
        )
        # predictor的input_dim应该是embed_dim + davf_feature_dim
        # 64 + 128 = 192
        first_layer = model.predictor.mlp[0]
        assert first_layer.in_features == 64 + 128

    def test_forward_with_davf(self):
        """带DAVF的forward正常执行"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            num_classes=4,
            use_davf=True,
            davf_config={"feature_dim": 64, "num_genes": 100},
        )
        batch = {
            "sequence": torch.randint(0, 21, (2, 100)),
            "ptm_types": torch.randint(0, 5, (2, 100)),
            "ptm_gene_names": [["AKT1", "TP53"], ["MAPK1"]],
            "ptm_type_names": [["Phosphorylation", "Ubiquitination"], ["Phosphorylation"]],
        }
        output = model(batch)
        assert "logits" in output
        assert output["logits"].shape == (2, 4)
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_architectures_davf.py -v
```

**Step 3: 修改architectures.py**

在PTM2CellNet.__init__中新增DAVF分支:
- 新增`use_davf`参数
- 新增`davf_config`参数
- 当`use_davf=True`时, 创建DAVFInferenceModule和PTMDirectionMapper
- predictor的input_dim变为`embed_dim + davf_feature_dim`

在PTM2CellNet.forward中:
- 如果use_davf且batch中有PTM基因信息, 执行DAVF推理
- 将DAVF特征与pooled特征concat后送入predictor

**Step 4: 运行全部测试**

```bash
pytest tests/unit/test_architectures_davf.py -v
pytest tests/ -x --timeout=60
```

**Step 5: Commit**

```bash
git add src/models/architectures.py tests/unit/test_architectures_davf.py
git commit -m "feat: integrate DAVF cascade fusion into PTM2CellNet architecture"
```

---

## Task 6: 更新models/__init__.py和配置

**Files:**
- Modify: `src/models/__init__.py`
- Create: `configs/davf_integration.yaml`

**Step 1: 更新__init__.py导出**

在`src/models/__init__.py`中添加DAVF相关导出。

**Step 2: 创建配置文件**

```yaml
# configs/davf_integration.yaml
model:
  encoder_type: "transformer"
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.1
  num_classes: 4
  task_type: "classification"
  
  # DAVF集成配置
  use_davf: true
  davf:
    state_space: "scvi_latent"
    checkpoint_path: "checkpoints/latent_davf_ibd_norman/best_model.pt"
    scvi_model_path: "checkpoints/scvi/ibd_norman_model/model.pt"
    geneformer_path: "checkpoints/geneformer_embeddings.pt"
    feature_dim: 128
    freeze: true
    num_genes: 5000
    num_inference_steps: 10

data:
  max_sequence_length: 1000
  ptm_types:
    - Phosphorylation
    - Ubiquitination
    - Acetylation
    - Methylation
    - Sumoylation
    - Succinylation

training:
  batch_size: 32
  epochs: 50
  lr: 0.001
  weight_decay: 0.01
```

**Step 3: Commit**

```bash
git add src/models/__init__.py configs/davf_integration.yaml
git commit -m "feat: add DAVF exports and integration config"
```

---

## Task 7: 端到端集成测试

**Files:**
- Create: `tests/integration/test_davf_integration.py`

**Step 1: 编写集成测试**

```python
# tests/integration/test_davf_integration.py
"""Integration tests for DAVF pipeline in PTM2CellNet."""
import pytest
import torch
from src.models.architectures import PTM2CellNet
from src.models.ptm_direction_mapper import PTMDirectionMapper
from src.models.davf_inference import DAVFInferenceModule, DAVFInferenceConfig


class TestDAVFIntegrationPipeline:
    """端到端集成测试(不依赖checkpoint文件)。"""

    def test_ptm_mapper_to_davf_inference(self):
        """PTMDirectionMapper → DAVFInferenceModule流程"""
        mapper = PTMDirectionMapper(
            num_genes=100,
            gene_name_list=[f"GENE_{i}" for i in range(100)],
        )
        
        result = mapper.map_ptm_to_perturbation(
            ptm_types=["Phosphorylation", "Ubiquitination"],
            gene_names=["GENE_5", "GENE_10"],
        )
        
        assert result["gene_ids"][0, 0].item() == 5
        assert result["gene_ids"][0, 1].item() == 10
        assert result["directions"][0, 0].item() == 2  # OE
        assert result["directions"][0, 1].item() == 0  # KO
        assert result["attention_mask"].sum().item() == 2.0

    def test_full_model_forward_without_checkpoint(self):
        """完整模型forward(DAVF无checkpoint时输出零特征)"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            num_classes=4,
            use_davf=True,
            davf_config={
                "feature_dim": 64,
                "num_genes": 100,
                "checkpoint_path": "/nonexistent/path.pt",
            },
        )
        batch = {
            "sequence": torch.randint(0, 21, (2, 50)),
            "ptm_types": torch.randint(0, 5, (2, 50)),
            "ptm_gene_names": [["AKT1"], ["MAPK1", "TP53"]],
            "ptm_type_names": [["Phosphorylation"], ["Phosphorylation", "Ubiquitination"]],
        }
        output = model(batch)
        assert output["logits"].shape == (2, 4)

    def test_gradient_flow_with_davf_frozen(self):
        """DAVF冻结时梯度仅流经projection和predictor"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=21,
            embed_dim=64,
            num_classes=4,
            use_davf=True,
            davf_config={
                "feature_dim": 64,
                "num_genes": 100,
                "freeze": True,
                "checkpoint_path": "/nonexistent/path.pt",
            },
        )
        batch = {
            "sequence": torch.randint(0, 21, (2, 50)),
            "ptm_types": torch.randint(0, 5, (2, 50)),
            "ptm_gene_names": [["AKT1"], ["MAPK1"]],
            "ptm_type_names": [["Phosphorylation"], ["Phosphorylation"]],
        }
        output = model(batch)
        loss = output["logits"].sum()
        loss.backward()
        
        # Predictor应有梯度
        for p in model.predictor.parameters():
            if p.requires_grad:
                assert p.grad is not None
```

**Step 2: 运行集成测试**

```bash
pytest tests/integration/test_davf_integration.py -v
```

**Step 3: Commit**

```bash
git add tests/integration/test_davf_integration.py
git commit -m "test: add DAVF integration tests"
```

---

## Task 8: 更新requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: 添加新依赖**

```
scvi-tools>=1.0
anndata>=0.10
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add scvi-tools and anndata dependencies for DAVF"
```

---

## 执行顺序总结

| Task | 内容 | 依赖 |
|------|------|------|
| 1 | 提取DAVF源文件 | 无 |
| 2 | 解压预训练模型 | 无 |
| 3 | PTMDirectionMapper | Task 1 |
| 4 | DAVFInferenceModule | Task 1, 2 |
| 5 | 修改PTM2CellNet架构 | Task 3, 4 |
| 6 | 更新init和配置 | Task 5 |
| 7 | 集成测试 | Task 5, 6 |
| 8 | 更新依赖 | 无 |

Task 1和2可并行执行。Task 3和4在Task 1完成后可并行。
