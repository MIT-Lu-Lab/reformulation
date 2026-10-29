"""Sphinx configuration.

autodoc imports what it documents, so this build needs jax and numpy (for
`pria`) and numpy and scipy (for `harness` and `baselines`).  It deliberately
does NOT need scipy for `pria` itself: that separation is what keeps the
library's install requirements to numpy and jax, and a docs build that quietly
started importing scipy from `pria` would be the first sign it had regressed.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))               # harness, baselines
sys.path.insert(0, str(_ROOT / "pria-jax"))  # pria

project = "jrp"
author = "Jacob Dentes"
copyright = "2026, Jacob Dentes"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# The docstrings are in Sphinx's native reST field style (:param:, :return:), so
# napoleon is here only to tolerate the occasional Google-style block.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "jrp"
html_static_path = []
