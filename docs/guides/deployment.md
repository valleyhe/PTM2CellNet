# 部署指南

## 概述

本指南介绍如何将PTM2CellNet部署到生产环境。

## Docker部署 (推荐)

### 1. 构建镜像

```bash
docker compose build
```

### 2. 启动服务

```bash
# 启动API服务
docker compose up -d

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f api
```

### 3. 验证部署

```bash
# 存活探针（始终 200，仅表示进程在跑）
curl http://localhost:8000/api/v1/live

# 就绪探针（模型加载后 200，未加载时 503；Docker HEALTHCHECK 用这个）
curl http://localhost:8000/api/v1/ready

# 综合健康（始终 200，通过 status 字段反映模型加载情况；勿用于流量判断）
curl http://localhost:8000/api/v1/health

# 预测测试
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "ptm_sites": [{"position": 10, "type": "phosphorylation"}]
  }'
```

## 生产环境配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PTM2CELLNET_CHECKPOINT` | 模型权重路径（`MODEL_PATH` 仅作兼容别名） | `outputs/models/best_model.pt` |
| `PTM2CELLNET_CONFIG` | 模型配置路径；未设置时可从 checkpoint sibling config 发现 | — |
| `PTM2CELLNET_API_PREFIX` | API 路由前缀 | `/api/v1` |
| `PTM2CELLNET_API_KEY` | 生产环境 API key；未设置时不启用兼容认证中间件 | — |
| `PTM2CELLNET_RATE_LIMIT_RPM` | 每进程每分钟请求上限；`0` 表示关闭 | `600` |
| `HF_ENDPOINT` | Hugging Face镜像 | `https://hf-mirror.com` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 扩缩容

> ⚠️ **TD-M08 进程模型（重要）**：镜像与 `configs/production.yaml` 默认 **单 worker**
> （`uvicorn --workers 1`）。速率限制（`PTM2CELLNET_RATE_LIMIT_RPM`）与 Prometheus
> `/metrics` 均为**进程内语义**：
> - 单 worker：限流与指标即全局值，语义一致；
> - 多 worker / 多副本：每个进程**独立**计数与采集，限流总量会被进程数放大、
>   `/metrics` 按进程分片（多副本场景 K8s 按 Pod 抓取可接受，多 worker 单 Pod
>   抓取 `/metrics` 会看到分片后的值）。
>
> 需要多进程/多副本扩展时，请先接入 Redis 或网关级共享限流（方案 A），再上调
> `--workers` 或副本数。当前默认配置下：

```bash
# 增加副本数（每个副本独立加载模型、独立限流计数）
docker compose up -d --scale api=3

# 使用负载均衡
# 需要配置nginx或其他反向代理
```

## Kubernetes部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ptm2cellnet
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ptm2cellnet
  template:
    metadata:
      labels:
        app: ptm2cellnet
    spec:
      containers:
      - name: api
        image: ptm2cellnet:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /api/v1/live   # 存活探针：进程在跑即 200
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/ready  # 就绪探针：模型加载后 200，未加载 503
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: ptm2cellnet-service
spec:
  selector:
    app: ptm2cellnet
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 跨尺度推理端点（cross-scale）

跨尺度模型（`CrossScalePTM2CellNet`，由 `scripts/train_cross_scale.py --output` 产出的
artifact 目录）通过独立的三个端点在线服务。它与标准 `/api/v1/predict` 链路
**完全独立**：独立状态槽、独立初始化，两类模型可在同一进程并存，互不干扰。

### 端点一览

| 端点 | 用途 | 关键行为 |
|---|---|---|
| `POST /api/v1/cross-scale/initialize` | 加载跨尺度 artifact | 校验 schema/词表/权重（`strict_assets=false` 仅放宽 best→last checkpoint 回退，其余校验不变） |
| `POST /api/v1/cross-scale/predict` | 单样本预测 | `sequence` 或 `embedding_ref` 二选一；二者皆无/皆有 → 400 |
| `POST /api/v1/cross-scale/batch_predict` | 批量预测 | `samples[]`（1–64，空列表 422）+ `batch_size`（1–64）；`fail_fast=false` 时逐样本收集 errors |

### initialize 关键字段

- `artifact_path`（必填）：`train_cross_scale.py` 产物目录（含 manifest/config/
  checkpoint/label 词表）。路径穿越被拒绝（403）。
- `graph_ref`（可选）：服务端 NPZ 默认图路径，需含 `cell_edge_index`（必需）、
  `signal_edge_index`、`signal_gene_map`。模型 `CellGraphCompassHead` 硬性要求
  显式 cell graph——在线请求无法内嵌大图，因此由 initialize 登记为默认图；
  单样本 embedding NPZ 自带图时优先使用。两者皆无且请求含序列/embedding
  需要图时 → 显式 400，不静默。
- `device` / `max_batch_size`：常规配置项。

### predict 关键字段

- `sequence`：原始氨基酸序列（服务端做 PTM 位点解析与词表映射，越界 400）。
- `embedding_ref` + `sample_index`：预计算 embedding NPZ 的服务端路径与行号，
  需含 `{backbone}_embeddings` 数组。
- `ptm_sites`：与标准端点相同的 `PTMSite` 列表（`position`/`type`/`gene_symbol`）。
- `include_delta_expression`：默认关闭；开启后返回 `num_cell_genes` 维
  `delta_expression` 向量。
- `allow_uniform_signal_map`（默认 `false`）：缺少可用 `signal_gene_map` 时的
  **显式工程选择**——按当前请求蛋白质节点数构造均匀映射（signal→gene 无信息
  退化），并在响应 `fallback_flags.signal_map_uniform=true` 标记。默认关闭时
  缺少映射直接 400。禁止隐式随机 fallback。

### 响应要点

每个 prediction 含 `cell_state`/`confidence`/`probabilities`/`cell_state_logits`、
`ptm_sites_applied`、`fallback_flags`（全部显式布尔，无静默降级）与
`provenance`（artifact manifest digest、模型来源等）。批量响应含
`summary{total,succeeded,failed}` 与逐样本 `errors[{index,sample_id,detail}]`。

> 与标准链路一致：跨尺度端点同样受 API key 门禁（401/403）、速率限制与
> 请求体大小限制约束；未初始化时返回 503。

## 监控

### 健康检查端点

| 端点 | 用途 | 行为 |
|---|---|---|
| `GET /api/v1/live` | **存活探针**（liveness） | 进程在跑即返回 200；用于判断是否需要重启容器 |
| `GET /api/v1/ready` | **就绪探针**（readiness） | 模型加载后 200，未加载时 **503**；用于 K8s `readinessProbe` / Docker `HEALTHCHECK` / 负载均衡流量接入 |
| `GET /api/v1/health` | 综合健康（遗留/兼容） | 始终 200，通过 `status` 字段（`healthy`/`unhealthy`）与 `model_loaded` 反映加载情况；**勿用于流量判断** |
| `GET /api/v1/model_info` | 模型信息 | 需模型已加载，否则 503 |

> ⚠️ **常见误用**：把 `livenessProbe`/`readinessProbe` 都指向 `/health`。由于 `/health` 在模型未加载时也返回 200，会把未就绪实例误判为健康，导致流量打到未加载模型的实例。**就绪判断请用 `/ready`。**

### 日志

```bash
# 查看API日志
docker compose logs -f api

# 日志位置
# 容器内: /app/outputs/logs/api.log
```

## 安全考虑

1. **API认证**: 在生产环境中启用API密钥或OAuth
2. **HTTPS**: 使用反向代理(nginx)配置TLS
3. **速率限制**: 配置请求频率限制防止滥用。`PTM2CELLNET_RATE_LIMIT_RPM` 为
   **每进程**上限（单 worker 部署下即全局上限）；多 worker/多副本部署时各进程
   独立计数，需在 Redis/网关层实施共享限流（见「扩缩容」节）
4. **输入验证**: 所有输入经过Pydantic验证
