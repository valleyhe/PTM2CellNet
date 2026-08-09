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

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = []
html_title = f"{project} {release} API Documentation"
