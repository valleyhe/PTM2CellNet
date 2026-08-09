# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
project = "PTM2CellNet"
copyright = "2026, PTM2CellNet Team"
author = "PTM2CellNet Team"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# Add the repository root to sys.path so autodoc can import the documented
# ``src.*`` namespace used by the API reference pages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# TD-M03（2026-08-09）: 包级 __init__ 通过 ``from .X import Y`` 重导出
# 符号（如 ``src.models.DAVFInferenceConfig``），且模块定义了 ``__all__``。
# Sphinx autodoc 在此结构下会把重导出类成员以两条路径注册
# （``src.models.DAVFInferenceConfig.attr`` 与其 canonical 模块路径），产生
# 已知的 "duplicate object description" 噪音——两条路径指向同一成员，文档
# 内容完全一致。该警告无 type 参数（Sphinx 的 suppress_warnings 无法覆盖
# type=None 的警告），故在此用 logging.Filter 按消息文本精确过滤，仅影响
# 该已知噪音类别，其他警告照常报告。若未来需要严格单路径命名空间，可为
# 每个子模块建立独立 rst 页面（见修复报告 TD-M03 方案）。
import logging as _logging


class _DuplicateObjectFilter(_logging.Filter):
    """过滤 autodoc 对重导出符号成员的双路径注册噪音。"""

    def filter(self, record: _logging.LogRecord) -> bool:
        return "duplicate object description" not in record.getMessage()


for _logger_name in ("sphinx.sphinx.domains.python", "sphinx.sphinx.ext.autodoc"):
    _logging.getLogger(_logger_name).addFilter(_DuplicateObjectFilter())

# 防止 autodoc 导入 src.api.app 时应用日志（如 CORS 安全提示）混入
# Sphinx 构建输出。
_logging.getLogger("src").setLevel(_logging.CRITICAL)

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = [
    "_build",
    "archive/**",
    "Thumbs.db",
    ".DS_Store",
    "**/*.pyc",
    "**/__pycache__",
]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# TD-M03: dataclass 的 ``Attributes:`` 节若转为 ``.. attribute::`` 指令，
# 会与 autodoc ``:members:`` 递归的 ``py:attribute`` 指令嵌套冲突
# （Unexpected indentation）。改用 ``:ivar:`` 字段形式，两者可共存。
napoleon_use_ivar = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = []
html_title = f"{project} {release} API Documentation"
