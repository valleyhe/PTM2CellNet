# 快速开始

## 5分钟体验PTM2CellNet

### 1. 准备环境

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 使用内置示例数据运行预测
python scripts/predict.py --sequence "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
```

### 3. 查看API文档

启动API服务器：
```bash
uvicorn src.api.app:app --reload
```

访问 http://localhost:8000/docs 查看交互式API文档。

## 训练你的第一个模型

### 使用示例数据

```bash
# 准备数据
python scripts/prepare_data.py

# 训练模型 (使用默认配置)
python scripts/train.py

# 评估模型
python scripts/evaluate.py --model outputs/models/best_model.pt
```

### 使用自定义数据

1. 准备CSV文件，包含以下列：
   - `id`: 样本唯一标识
   - `sequence`: 蛋白质氨基酸序列
   - `ptm_sites`: PTM位点JSON字符串
   - `cell_state`: 细胞状态标签

2. 运行训练：
```bash
python scripts/train.py --data your_data.csv --config configs/default.yaml
```

## 使用Docker部署

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f api

# 测试API
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MVLSPADKTNVKAA", "ptm_sites": []}'
```

## 下一步

- 阅读 [安装指南](installation.md) 了解详细安装选项
- 阅读 [数据整合指南](data_integration.md) 了解如何准备训练数据
- 阅读 [训练指南](training.md) 了解模型调优
- 阅读 [部署指南](deployment.md) 了解生产环境配置
