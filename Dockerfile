# PTM2CellNet 生产环境 Docker 镜像
FROM python:3.10-slim

LABEL maintainer="PTM2CellNet Team"
LABEL description="蛋白质PTM与细胞状态预测系统"

# 设置环境变量
# P0-2: 统一使用 PTM2CELLNET_CHECKPOINT / PTM2CELLNET_CONFIG 作为正式变量名，
# 与 src/api/app.py 自动加载逻辑保持一致。MODEL_PATH 已弃用（代码仍兼容并告警）。
ENV PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PTM2CELLNET_CHECKPOINT=/app/outputs/models/best_model.pt \
    PTM2CELLNET_CONFIG=/app/outputs/models/best_model.config.yaml

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
# P1-2: 默认只安装 core 依赖（最小训练/推理/API 闭环），保持镜像精简且对
# CPU-only / 干净构建机友好。如需预训练/Mamba/分析能力，通过 build arg 打开：
#   docker build --build-arg INSTALL_PRETRAINED=1 ...
COPY requirements-core.txt requirements-pretrained.txt \
     requirements-mamba.txt requirements-analysis.txt \
     requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-core.txt

# 可选能力 build arg（默认关闭）。叠加在 core 之上，按需安装。
ARG INSTALL_PRETRAINED=0
ARG INSTALL_MAMBA=0
ARG INSTALL_ANALYSIS=0
RUN if [ "$INSTALL_PRETRAINED" = "1" ]; then \
        pip install --no-cache-dir -r requirements-pretrained.txt; \
    fi ; \
    if [ "$INSTALL_MAMBA" = "1" ]; then \
        pip install --no-cache-dir -r requirements-mamba.txt; \
    fi ; \
    if [ "$INSTALL_ANALYSIS" = "1" ]; then \
        pip install --no-cache-dir -r requirements-analysis.txt; \
    fi

# 复制源代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 复制 demo 模型权重与配置，使镜像开箱即用 /api/v1/ready 直接通过。
# 注意：这是合成数据训练的 demo 模型（见 outputs/models/README.md），
# 仅用于工程链路验证，不可用于真实生物学预测。生产部署如需挂载真实模型，
# 可挂载 outputs/models 目录并覆盖 PTM2CELLNET_CHECKPOINT/CONFIG。
COPY outputs/models/best_model.pt ./outputs/models/best_model.pt
COPY outputs/models/best_model.config.yaml ./outputs/models/best_model.config.yaml
COPY outputs/models/artifact_manifest.json ./outputs/models/artifact_manifest.json

# 创建必要目录
RUN mkdir -p outputs/models outputs/logs outputs/results \
    .cache/huggingface

# SEC-05: Docker 安全加固 —— 创建非 root 用户并确保 /app 目录权限正确。
# chown 必须在 USER 切换之前以 root 身份执行，否则非 root 用户无法修改属主。
# chown -R /app 同时覆盖 /app/outputs（outputs/models|logs|results 均被引用），
# 因此容器以非 root 身份运行时，应用对输出目录具备读写权限。
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# 以非 root 用户运行容器，符合最小权限原则。
USER appuser

# 暴露API端口
EXPOSE 8000

# 健康检查使用 readiness 端点 /api/v1/ready：模型未加载时返回 503，
# 避免把流量打到尚未完成自动初始化的实例（P0-2）。
# start-period 设为 30s，给模型自动加载（含可选 variant workflow 初始化）留出余量。
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/ready || exit 1

# 启动FastAPI服务（app.py 模块级别导出 app 实例）
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
