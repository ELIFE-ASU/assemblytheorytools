# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath('../../'))  # Points to project root

# Importing the package applies matplotlib rcParams at module scope, so pin a
# headless backend before autodoc imports anything.
os.environ.setdefault('MPLBACKEND', 'Agg')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'assemblytheorytools'
copyright = '2025, Louie Slocombe et al.'
author = 'Louie Slocombe et al.'
# Single-sourced from the installed package's metadata (see pyproject.toml).
release = _pkg_version('assemblytheorytools')
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.coverage',  # Report undocumented objects via `make coverage`
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc options ---------------------------------------------------------
# Nothing is mocked: autodoc imports the real package, so a broken or missing
# runtime dependency surfaces as a build warning (an error under `-W`) instead
# of silently emitting an empty module page. Install the docs dependencies with
# `pip install -e ".[docs]"` from the project root.
autodoc_mock_imports = []

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Keep annotations in the signature; the NumPy-style docstrings carry their own
# parameter types, so duplicating them in the description is noise.
autodoc_typehints = 'signature'
autodoc_preserve_defaults = True

# -- Napoleon options --------------------------------------------------------
# NumPy style only: Google-style parsing is disabled so that a Google-style
# section header is surfaced as a malformed docstring instead of silently
# rendering.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
# Render class `Attributes` sections as :ivar: fields. Without this, autodoc
# also emits an attribute directive per dataclass field, and the two collide as
# duplicate object descriptions.
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
