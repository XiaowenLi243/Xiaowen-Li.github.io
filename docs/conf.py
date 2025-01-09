import tomli
from pathlib import Path
from datetime import datetime

import hsic_optimization

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

with open("../pyproject.toml", "rb") as fd:
    config = tomli.load(fd)

project = config["project"]["name"].replace("_", " ").title()
copyright = f"{datetime.now().year}, {config['project']['authors'][0]['name']}"
author = config["project"]["authors"][0]["name"]

# The full version, including alpha/beta/rc tags
release = hsic_optimization.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_nb", "autoapi.extension"]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------
autoapi_dirs = ["../src/hsic_optimization"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "undoc-members",
]
autoapi_ignore = ["*plugin.py", "*vendor*"]

autodoc_typehints = "description"

nb_execution_mode = "off"

myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 2

# -- Additional code ---------------------------------------------------------

# copy README locally, adjusting links and title
readme_txt = Path("../README.md").read_text()
readme_lines = readme_txt.splitlines()
readme_txt = "\n".join(["# Main instructions"] + readme_lines[1:])
readme_txt = readme_txt.replace("(doc/", "(")
Path("./01_README.md").write_text(readme_txt)
