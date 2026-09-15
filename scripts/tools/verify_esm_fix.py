#!/usr/bin/env python3
"""
ESM2 Tokenizer修复全面验证脚本
验证所有修改是否正确生效
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import pandas as pd

print("=" * 60)
print("ESM2 Tokenizer 修复验证")
print("=" * 60)

# Check 1: ESM2Encoder.tokenizer exists
print("\n[1/6] 验证 ESM2Encoder.tokenizer 存在...")
from src.models.pretrained_encoders import ESM2Encoder

encoder = ESM2Encoder(model_size="8M", freeze=True)
assert hasattr(encoder, "tokenizer"), "❌ ESM2Encoder 缺少 tokenizer 属性"
assert encoder.tokenizer is not None, "❌ tokenizer 为 None"
print("✅ ESM2Encoder.tokenizer 存在")

# Check 2: tokenize() method works
print("\n[2/6] 验证 tokenize() 方法输出...")
result = encoder.tokenize(["ACDEFGHIKL"])
assert "input_ids" in result, "❌ tokenize() 输出缺少 input_ids"
assert "attention_mask" in result, "❌ tokenize() 输出缺少 attention_mask"
assert result["input_ids"].shape[1] == 12, f"❌ input_ids 长度错误: {result['input_ids'].shape[1]} (期望12=10+2)"
print(f"✅ tokenize() 输出正确: input_ids shape={result['input_ids'].shape}")

# Check 3: ESMTokenizedDataset works
print("\n[3/6] 验证 ESMTokenizedDataset...")
from src.data.datasets import ESMTokenizedDataset

test_df = pd.DataFrame(
    {
        "sequence": ["ACDEFGHIKL"],
        "ptm_sites": ['[{"position": 3, "type": "Phosphorylation"}]'],
        "cell_state": ["Activated"],
    }
)
dataset = ESMTokenizedDataset(df=test_df, tokenizer=encoder.tokenizer, config={})
sample = dataset[0]
assert "input_ids" in sample, "❌ ESMTokenizedDataset 输出缺少 input_ids"
assert "ptm_mask" in sample, "❌ ESMTokenizedDataset 输出缺少 ptm_mask"
assert sample["input_ids"].shape[0] == sample["ptm_mask"].shape[0], "❌ input_ids 和 ptm_mask 长度不一致"
assert sample["ptm_mask"][3] == 1.0, f"❌ PTM 位置对齐错误: ptm_mask[3]={sample['ptm_mask'][3]} (期望1.0)"
print(f"✅ ESMTokenizedDataset 工作正常，PTM 位置对齐正确")

# Check 4: PTMDataModule accepts tokenizer
print("\n[4/6] 验证 PTMDataModule 接受 tokenizer 参数...")
from src.data.lightning_datamodule import PTMDataModule

data_module = PTMDataModule(train_df=test_df, val_df=test_df, test_df=test_df, config={}, tokenizer=encoder.tokenizer)
data_module.setup("fit")
assert data_module.train_dataset is not None, "❌ train_dataset 未初始化"
assert isinstance(data_module.train_dataset, ESMTokenizedDataset), "❌ 未使用 ESMTokenizedDataset"
print("✅ PTMDataModule 正确使用 ESMTokenizedDataset")

# Check 5: PTM2CellNet.forward() handles input_ids
print("\n[5/6] 验证 PTM2CellNet.forward() 处理 input_ids...")
from src.models.architectures import PTM2CellNet

model = PTM2CellNet(encoder_type="esm2_8M", num_classes=2, freeze_encoder=True)
model.eval()
batch = {
    "input_ids": sample["input_ids"].unsqueeze(0),
    "attention_mask": sample["attention_mask"].unsqueeze(0),
    "ptm_mask": sample["ptm_mask"].unsqueeze(0),
    "ptm_types": sample["ptm_types"].unsqueeze(0),
}
with torch.no_grad():
    output = model(batch)
assert "logits" in output, "❌ 模型输出缺少 logits"
assert output["logits"].shape == (1, 2), f"❌ logits shape 错误: {output['logits'].shape}"
print("✅ PTM2CellNet.forward() 正确处理 input_ids")

# Check 6: input_ids range validation
print("\n[6/6] 验证 input_ids 值范围...")
input_ids = sample["input_ids"]
min_id = input_ids.min().item()
max_id = input_ids.max().item()
vocab_size = encoder.tokenizer.vocab_size
assert min_id >= 0, f"❌ input_ids 最小值 {min_id} < 0"
assert max_id < vocab_size, f"❌ input_ids 最大值 {max_id} >= vocab_size {vocab_size}"
print(f"✅ input_ids 范围正确: [{min_id}, {max_id}] < vocab_size={vocab_size}")

print("\n" + "=" * 60)
print("✅ 所有验证通过！ESM2 tokenizer 修复成功生效")
print("=" * 60)
