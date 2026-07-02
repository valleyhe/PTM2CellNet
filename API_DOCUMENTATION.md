# PTM2CellNet API 文档

## 1. API 概述

PTM2CellNet API 基于 FastAPI，提供蛋白质序列与 PTM（翻译后修饰）驱动的细胞状态预测能力。

- API 版本：`v1`
- 响应格式：`application/json`
- 文档入口：`/docs`（Swagger UI）、`/redoc`

## 2. 基础 URL

- 本地开发：`http://127.0.0.1:8000`
- API 前缀：`/api/v1`
- 完整基础地址：`http://127.0.0.1:8000/api/v1`

启动示例：

```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

## 3. 端点说明

### 3.1 单样本预测 `POST /api/v1/predict`

用途：对单条蛋白质序列进行细胞状态预测。

请求体示例：

```json
{
  "sequence": "ACDEFGHIKLMNPQRSTVWY",
  "ptm_sites": [
    {
      "position": 5,
      "type": "phosphorylation",
      "amino_acid": "F"
    },
    {
      "position": 12,
      "type": "acetylation"
    }
  ]
}
```

成功响应示例（`200`）：

```json
{
  "cell_state": "proliferation",
  "confidence": 0.8732,
  "probabilities": {
    "proliferation": 0.8732,
    "differentiation": 0.0711,
    "apoptosis": 0.0399,
    "quiescence": 0.0158
  },
  "processing_time_ms": 12.47
}
```

字段说明：

- `sequence`：蛋白质氨基酸序列
- `ptm_sites[].position`：PTM 位点（从 1 开始）
- `ptm_sites[].type`：PTM 类型（如 `phosphorylation`、`acetylation`）

---

### 3.2 批量预测 `POST /api/v1/predict/batch`

用途：批量提交样本并返回逐样本预测结果。

兼容路径：`POST /api/v1/batch_predict`

请求体示例：

```json
{
  "samples": [
    {
      "sequence": "ACDEFGHIKL",
      "ptm_sites": []
    },
    {
      "sequence": "LMNPQRSTVWY",
      "ptm_sites": [
        {
          "position": 3,
          "type": "acetylation"
        }
      ]
    }
  ]
}
```

成功响应示例（`200`）：

```json
{
  "predictions": [
    {
      "cell_state": "proliferation",
      "confidence": 0.6123,
      "probabilities": {
        "proliferation": 0.6123,
        "differentiation": 0.2110,
        "apoptosis": 0.1302,
        "quiescence": 0.0465
      },
      "processing_time_ms": 8.91
    },
    {
      "cell_state": "differentiation",
      "confidence": 0.7021,
      "probabilities": {
        "proliferation": 0.1202,
        "differentiation": 0.7021,
        "apoptosis": 0.1127,
        "quiescence": 0.0649
      },
      "processing_time_ms": 9.34
    }
  ],
  "total_processing_time_ms": 18.73,
  "sample_count": 2
}
```

---

### 3.3 健康检查 `GET /api/v1/health`

用途：检查服务运行状态和模型加载状态。

成功响应示例（`200`）：

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "timestamp": "2026-03-30T10:23:59.123456"
}
```

---

### 3.4 模型信息 `GET /api/v1/model/info`

用途：获取当前已加载模型的元信息。

兼容路径：`GET /api/v1/model_info`

成功响应示例（`200`）：

```json
{
  "model_name": "PTM2CellNet",
  "model_version": "1.0.0",
  "encoder_type": "esm2_8m",
  "embed_dim": 320,
  "num_classes": 4,
  "cell_states": [
    "proliferation",
    "differentiation",
    "apoptosis",
    "quiescence"
  ],
  "supported_ptm_types": [
    "phosphorylation",
    "acetylation",
    "ubiquitination",
    "methylation"
  ]
}
```

---

### 3.5 变体预测 `POST /api/v1/predict/variant`

用途：基于蛋白质变体的 HGVS 表达式预测细胞状态变化，并评估对 PTM 与信号通路的潜在影响。

请求体字段说明：

- `hgvs`：必填，HGVS 变体表示法，例如 `BRAF:p.V600E`
- `sequence`：可选，蛋白质序列；未提供时可结合 `uniprot_id` 自动拉取
- `uniprot_id`：可选，用于从 UniProt 获取蛋白质序列
- `include_pathways`：可选，是否返回通路影响分析结果，默认 `true`

请求体示例：

```json
{
  "hgvs": "BRAF:p.V600E",
  "uniprot_id": "P15056",
  "include_pathways": true
}
```

成功响应示例（`200`）：

```json
{
  "variant": {
    "hgvs": "BRAF:p.V600E",
    "gene_symbol": "BRAF",
    "position": 600,
    "ref_aa": "V",
    "alt_aa": "E"
  },
  "ptm_effects": [
    {
      "ptm_type": "phosphorylation",
      "wildtype_prob": 0.42,
      "mutant_prob": 0.78,
      "delta_prob": 0.36,
      "effect": "gain"
    },
    {
      "ptm_type": "ubiquitination",
      "wildtype_prob": 0.31,
      "mutant_prob": 0.19,
      "delta_prob": -0.12,
      "effect": "loss"
    }
  ],
  "pathway_impacts": [
    {
      "pathway_name": "MAPK signaling",
      "activity_change": 0.67,
      "confidence": "high",
      "key_genes": [
        "BRAF",
        "MAP2K1",
        "MAPK1"
      ]
    }
  ],
  "confidence": 0.72,
  "processing_time_ms": 48.15,
  "warnings": []
}
```

响应字段说明：

- `variant.hgvs`：输入的 HGVS 变体表示
- `variant.gene_symbol`：解析得到的基因符号
- `variant.position`：变体位点（1-based）
- `variant.ref_aa`：参考氨基酸
- `variant.alt_aa`：突变后氨基酸
- `ptm_effects`：PTM 影响列表，包含各 PTM 类型在野生型与突变型之间的概率变化
- `pathway_impacts`：通路影响列表；当 `include_pathways=false` 时可能为空
- `confidence`：整体预测置信度
- `processing_time_ms`：请求处理耗时（毫秒）
- `warnings`：处理过程中的警告信息

## 4. 错误码说明

| 状态码 | 含义 | 常见触发场景 |
|---|---|---|
| `200` | 请求成功 | 预测/查询成功 |
| `400` | 业务参数错误 | 序列非法、长度超限、请求内容不满足业务约束 |
| `422` | 请求体校验失败 | 缺失必填字段、字段类型错误（Pydantic 校验失败） |
| `500` | 服务器内部错误 | 推理阶段发生未处理异常 |
| `503` | 服务不可用 | 模型未初始化或未完成加载 |

错误响应示例（`400`）：

```json
{
  "detail": "无效序列: 序列包含非法字符 X"
}
```

错误响应示例（`503`）：

```json
{
  "detail": "模型未初始化"
}
```

## 5. 调用建议

- 生产环境建议通过反向代理统一管理超时、限流与鉴权。
- 批量预测建议控制单次 `samples` 数量，避免单请求耗时过长。
- 对 `5xx` 错误建议使用指数退避重试；对 `4xx` 错误建议修正输入后重试。
