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

```bash
# 增加worker数量
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
3. **速率限制**: 配置请求频率限制防止滥用
4. **输入验证**: 所有输入经过Pydantic验证
