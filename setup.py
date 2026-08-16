"""
PTM2CellNet 安装脚本

依赖声明单一事实源（N02 修复，2026-08-16）：
    - ``requirements-core.txt`` 是运行时核心依赖的唯一权威声明，
      本文件的 ``install_requires`` 是其镜像，必须保持一致；
      ``tests/unit/test_dependency_contracts.py`` 自动对比，漂移即失败。
    - ``extras_require`` 与 ``requirements-{pretrained,mamba,analysis,docs,dev}.txt``
      一一对应（包名集合一致），按场景通过 ``pip install -e .[extra]`` 安装。
    - 最小核心原则：transformers / lightning / mamba 等重型依赖不进入
      ``install_requires``，只出现在对应 extra 中。
"""

from setuptools import setup, find_packages

# 与 requirements-core.txt 保持一致（镜像，禁止单独修改；一致性由测试门禁）。
install_requires = [
    # Core data
    "numpy>=1.24,<3",  # D1: 上限放宽至 <3 对齐 requirements-core.txt（lock==2.4.3 实测兼容）
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    # Deep learning
    "torch>=2.0,<3",
    # Machine learning
    "scikit-learn>=1.0.0",
    "torchmetrics>=0.11.0",
    "networkx>=2.6.0",
    # Bioinformatics (lightweight)
    "biopython>=1.79",
    "datasketch>=1.5.0",
    # Visualization
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    # Config & tools
    "pyyaml>=5.4.0",
    "tqdm>=4.62.0",
    "h5py>=3.6.0",
    "requests>=2.31.0",
    "typing_extensions>=4.0.0",
    "setuptools>=68.0,<81",
    # Web API (FastAPI service)
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "pydantic>=1.10,<3",
    "starlette>=0.40.0",
    # joblib used by src/data/homology_splitter.py
    "joblib>=1.2.0",
    # safetensors used by src/utils/checkpoint_utils.py and src/data/validation.py
    "safetensors>=0.3.0",
]

_pretrained = [
    "transformers>=4.20.0",
    "sentencepiece>=0.2.0",
    "peft>=0.8.0",
    "einops>=0.6.0",
    "lightning>=2.5.0",
    # torchmetrics 已含于 install_requires（core 镜像）
    "tensorboard>=2.13.0",
    "esm>=3.0.0",
]
_mamba = [
    "mamba-ssm>=2.0.0",
    "causal-conv1d>=1.4.0",
    "lion-pytorch>=0.0.7",
    "einops>=0.6.0",
]
_analysis = [
    "anndata>=0.10,<0.12",
    "sspa>=0.2.0",
    "scvi-tools>=1.2.0",
    "zarr>=2,<3",
    "hgvs==1.5.7",
    "uniprot-id-mapper==1.1.5",
    "UniProtMapper>=0.1",
    "torch-geometric>=2.3.0",
]
_genki = [
    "torch-geometric>=2.3.0",
    "scvi-tools>=1.2.0",
]
# API 服务依赖已含于 install_requires（core）；保留空 extra 兼容历史
# ``pip install -e .[api]`` 命令（N02：版本以 requirements-core.txt 为准）。
_api = [
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "pydantic>=1.10,<3",
    "starlette>=0.40.0",
]
_docs = [
    "sphinx>=7.0.0",
    "myst-parser>=3.0.0",
    "sphinx-rtd-theme>=2.0.0",
]
_dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-timeout>=2.1.0",
    "types-requests>=2.31.0",
    "types-PyYAML>=6.0.0",
    "types-setuptools>=65.0.0",
    "pandas-stubs>=2.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

extras_require = {
    # ↔ requirements-pretrained.txt（在其之上叠加）
    "pretrained": _pretrained,
    # ↔ requirements-mamba.txt（在其之上叠加）
    "mamba": _mamba,
    # ↔ requirements-analysis.txt（在其之上叠加）
    "analysis": _analysis,
    # GenKI 图扰动后端（analysis 的子集，独立按需安装）
    "genki": _genki,
    # API 服务依赖已含于 install_requires（core）；保留空 extra 兼容历史
    # ``pip install -e .[api]`` 命令（N02：版本以 requirements-core.txt 为准）。
    "api": _api,
    # ↔ requirements-docs.txt
    "docs": _docs,
    # ↔ requirements-dev.txt
    "dev": _dev,
    "all": _pretrained + _mamba + _analysis + _api + _docs + _dev,
}

setup(
    name="ptm2cellnet",
    version="1.0.0",
    packages=find_packages(),
    author="PTM2CellNet Team",
    description="蛋白质PTM与细胞状态预测系统",
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ptm2cellnet-train=scripts.train:main",
            "ptm2cellnet-predict=scripts.predict:main",
        ],
    },
)
