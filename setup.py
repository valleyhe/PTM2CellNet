"""
PTM2CellNet 安装脚本

依赖说明：
    - ``install_requires``：运行核心训练/推理所需的最小依赖集。
    - ``extras_require``：按使用场景分组，可通过 ``pip install -e .[api]``
      等方式按需安装，避免一次性拉取全部重量级依赖。
"""

from setuptools import setup, find_packages

setup(
    name="ptm2cellnet",
    version="1.0.0",
    packages=find_packages(),
    author="PTM2CellNet Team",
    description="蛋白质PTM与细胞状态预测系统",
    python_requires=">=3.10",
    install_requires=[
        # 核心数据与科学计算
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        # 深度学习
        "torch>=1.10.0",
        # 机器学习
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        # 生物信息学
        "biopython>=1.79",
        # 可视化
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        # 配置与工具
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "h5py>=3.6.0",
        "requests>=2.26.0",
        "typing_extensions>=4.0.0",
        # setuptools<81 keeps the ``pkg_resources`` module available; it was
        # removed in setuptools 81, which scvi-tools (and other scientific
        # packages) still import at runtime.
        "setuptools>=68.0,<81",
    ],
    extras_require={
        # Web API 服务（FastAPI）
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.10,<3",
        ],
        # Lightning 训练框架
        "lightning": [
            "lightning>=2.5.0",
            "torchmetrics>=0.11.0",
            "tensorboard>=2.13.0",
        ],
        # 预训练蛋白质语言模型
        "pretrained": [
            "transformers>=4.20.0",
            "peft>=0.8.0",
            "einops>=0.6.0",
        ],
        # 高级单细胞 / 通路分析
        # anndata<0.12 与 scvi-tools>=1.2.0 联合约束（审计 P1-1）：旧版
        # scvi-tools 依赖被 anndata 0.11 移除的 SparseDataset，导致 import
        # 失败；scvi-tools 1.2.0+ 已改用 CSRDataset/CSCDataset。
        "analysis": [
            "anndata>=0.10,<0.12",
            "sspa>=0.2.0",
            "scvi-tools>=1.2.0",
            "zarr>=2,<3",  # anndata 0.11.x 在 zarr>=3 上 ImportError
            "hgvs==1.5.7",
            "uniprot-id-mapper==1.1.5",
        ],
        # GENKI 图神经网络 / 单细胞分析支持
        "genki": [
            "torch-geometric>=2.5.0",
            "scanpy>=1.9.0",
            "anndata>=0.8.0",
        ],
        # Mamba 编码器（可选高性能实现）
        "mamba": [
            "mamba-ssm>=2.0.0",
            "lion-pytorch>=0.0.7",
        ],
        # 全量安装（不含 dev）
        "all": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.10,<3",
            "lightning>=2.5.0",
            "torchmetrics>=0.11.0",
            "tensorboard>=2.13.0",
            "transformers>=4.20.0",
            "peft>=0.8.0",
            "einops>=0.6.0",
            "anndata>=0.10,<0.12",
            "sspa>=0.2.0",
            "scvi-tools>=1.2.0",
            "zarr>=2,<3",
            "hgvs==1.5.7",
            "uniprot-id-mapper==1.1.5",
            "torch-geometric>=2.5.0",
            "scanpy>=1.9.0",
            "mamba-ssm>=2.0.0",
            "lion-pytorch>=0.0.7",
            "torchvision>=0.11.0",
            "python-dotenv>=0.19.0",
        ],
        # 开发与测试
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "httpx>=0.24.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ptm2cellnet-train=scripts.train:main",
            "ptm2cellnet-predict=scripts.predict:main",
        ],
    },
)
