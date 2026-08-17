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
copyright = '2025-2026, Louie Slocombe et al.'
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
    'sphinx.ext.autosummary',  # Per-module summary tables on the API pages
    'sphinx.ext.intersphinx',  # Resolve types to the upstream project's docs
    'myst_parser',  # Narrative pages are Markdown; also enables `include`
]

templates_path = ['_templates']
exclude_patterns = []

# The narrative pages are MyST Markdown, the API stubs stay reStructuredText.
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# colon_fence lets directives be written as ::: blocks, which survives being
# viewed on GitHub; deflist is used by the configuration reference.
myst_enable_extensions = ['colon_fence', 'deflist']
# The included READMEs carry their own `#` title, so allow headings to start
# below the page title without a warning.
myst_heading_anchors = 3

# `sphinx.ext.autosectionlabel` is deliberately NOT enabled: the example pages
# include READMEs that repeat section titles ("It illustrates how to:"), which
# would collide as duplicate labels and fail the build under -W.

# -- Intersphinx -------------------------------------------------------------
# Without these, every `nx.Graph` / `np.ndarray` in a signature renders as inert
# text. Neither RDKit nor ASE publishes an objects.inv, so their types are listed
# in `nitpick_ignore_regex` below instead of being mapped here.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'networkx': ('https://networkx.org/documentation/stable', None),
}
# Do not fail the build when a docs site is unreachable; a network blip during a
# release build must not block it.
intersphinx_timeout = 30

# -- Autosummary -------------------------------------------------------------
# The API pages carry summary tables only; `automodule` still owns every object
# page, so stub generation stays off to avoid duplicate object descriptions.
autosummary_generate = False

# -- Nitpicky mode -----------------------------------------------------------
# On, so that a dead cross-reference written in prose -- `:func:`does_not_exist``
# -- fails the build instead of silently rendering as plain text. Without this,
# `fail_on_warning` gives false confidence: unresolved references are not
# warnings by default.
#
# The ignore list below covers the ~550 references that come from NumPy-style
# *type* strings rather than prose, and which cannot be resolved:
#
#   * Libraries that publish no objects.inv (RDKit, ASE, Pillow, pyvis,
#     pubchempy). These cannot be fixed from this end.
#   * Abbreviated aliases in docstrings (``nx.Graph``, ``np.ndarray``,
#     ``pd.DataFrame``) rather than the importable path.
#   * Fragments produced when napoleon splits a compound type on its commas,
#     so ``Union[nx.Graph, Chem.Mol]`` arrives as ``Union[nx.Graph`` plus
#     ``Chem.Mol]``. These are matched by the bracket/paren pattern.
#   * Prose type words such as "array-like" and "sequence".
#
# Deliberately NOT ignored: py:func, py:meth and unqualified py:obj targets, so
# an explicit role pointing at something that does not exist still fails.
nitpicky = True
nitpick_ignore_regex = [
    # Any fragment carrying a bracket or paren: a mangled compound type.
    ('py:class', r'.*[\[\]()].*'),
    # Third-party projects with no intersphinx inventory.
    ('py:class', r'(rdkit|Chem|ase|PIL|pyvis|pubchempy|dagviz)\..*'),
    ('py:exc', r'(rdkit|Chem|ase|PIL|pyvis|pubchempy|dagviz)\..*'),
    # Abbreviated module aliases used in docstring type strings.
    ('py:class', r'(nx|np|pd|plt|mpl)\..*'),
    # Prose type words.
    ('py:class', r'(array-like|array_like|sequence|iterable|scalar|'
                 r'file-like|path-like|optional|callable|dict-like)'),
    # Type variables, which have no documented target.
    ('py:class', r'.*\.T$'),
    ('py:obj', r'typing\..*'),
]

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
html_title = f'{project} {version}'

# Drives the theme's "Edit on GitHub" link and the version banner.
html_context = {
    'display_github': True,
    'github_user': 'ELIFE-ASU',
    'github_repo': 'assemblytheorytools',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
    'titles_only': False,
}
