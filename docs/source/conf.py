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
import os
import sys

autodoc_mock_imports = [
    "torch",
    "numpy",
    "pandas",
    "sklearn",
    "torchmetrics",
    "torchvision",
    "einops",
    "albumentations",
]

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "TorchFlare"
project_copyright = "2021, Atharva Phatak"
author = "Atharva Phatak"

# The full version, including alpha/beta/rc tags
with open("../../version.txt") as f:
    release = str(f.readline().strip())


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
# html_theme_options = {"collapse_navigation": False}
pygments_style = "friendly"
html_context = {
    "display_github": True,
    "github_user": "Atharva-Phatak",
    "github_repo": "torchflare",
}
html_logo = "_static/logo.png"
html_theme_options = {
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/Atharva-Phatak/torchflare/tree/main",
            "icon": "fab fa-github-square",
        }
    ],
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
