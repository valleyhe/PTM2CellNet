"""
PTM2CellNet 安装脚本

依赖说明：
    - ``install_requires``：运行核心训练/推理所需的最小依赖集。
    - ``extras_require``：按使用场景分组，可通过 ``pip install -e .[api]``
      等方式按需安装，避免一次性拉取全部重量级依赖。
"""

from setuptools import setup, find_packages

extras_require = {
    "pretrained": [
        "transformers>=4.30.0",
        "fair-esm>=2.0.0",
        "tokenizers>=0.13.0",
    ],
    "mamba": [
        "mamba-ssm>=2.0.0",
        "causal-conv1d>=1.0.0",
    ],
    "analysis": [
        "rpy2>=3.5.0",
        "sspa>=0.1.0",
        "anndata>=0.8.0",
        "UniProtMapper>=0.1",
    ],
    "genki": [
        "torch-geometric>=2.3.0",
        "scvi-tools>=1.0.0",
    ],
    "api": [
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
    ],
    "all": [
        "transformers>=4.30.0",
        "fair-esm>=2.0.0",
        "tokenizers>=0.13.0",
        "mamba-ssm>=2.0.0",
        "causal-conv1d>=1.0.0",
        "rpy2>=3.5.0",
        "sspa>=0.1.0",
        "anndata>=0.8.0",
        "UniProtMapper>=0.1",
        "torch-geometric>=2.3.0",
        "scvi-tools>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
    ],
}

setup(
    name="ptm2cellnet",
    version="1.0.0",
    packages=find_packages(),
    author="PTM2CellNet Team",
    description="蛋白质PTM与细胞状态预测系统",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0,<3",
        "numpy>=1.24,<2",
        "pandas>=1.3.0",
        "scikit-learn>=1.0",
        "scipy>=1.7.0",
        "transformers>=4.30.0",
        "lightning>=2.5.0",
        "pyyaml>=6.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ptm2cellnet-train=scripts.train:main",
            "ptm2cellnet-predict=scripts.predict:main",
        ],
    },
)
