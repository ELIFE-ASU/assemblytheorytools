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
    'myst_nb',  # Renders the protocol notebooks and the Markdown pages. It wraps
                # myst_parser, so do NOT also list 'myst_parser' here: registering
                # the MyST config values twice raises an ExtensionError.
]

templates_path = ['_templates']
exclude_patterns = []

# The narrative pages are MyST Markdown, the protocol pages are Jupyter
# notebooks, and the API stubs stay reStructuredText. MyST-NB owns both '.md'
# and '.ipynb': the plain 'markdown' parser is no longer registered once
# myst_parser is loaded via myst_nb, so mapping '.md' to 'myst-nb' is required,
# not optional. Ordinary Markdown falls back to the MyST parser transparently.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

# colon_fence lets directives be written as ::: blocks, which survives being
# viewed on GitHub; deflist is used by the configuration reference; dollarmath
# parses the $...$ and $$...$$ used for the equations on the theory page.
myst_enable_extensions = ['colon_fence', 'deflist', 'dollarmath']
# The included READMEs carry their own `#` title, so allow headings to start
# below the page title without a warning.
myst_heading_anchors = 3

# -- MyST-NB (notebook rendering) --------------------------------------------
# The four protocol pages are executed Jupyter notebooks committed with their
# outputs. They are rendered as stored and never executed by the docs build:
# they need external datasets, network access and a large amount of CPU time.
# Re-execute them with `jupyter nbconvert --execute` after editing, not here.
nb_execution_mode = 'off'
# Collapse each stream into a single block instead of one per print() call.
nb_merge_streams = True
# Standard-error text (e.g. the mzML parser's log lines in Protocol 4) is shown
# in the rendered output rather than turned into a build warning.
nb_output_stderr = 'show'

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
# Limit how long inventory downloads can delay a build.
intersphinx_timeout = 30

# Covers the warnings raised by an explicit ``:external:`` role -- an unknown
# inventory name, or an ambiguous match across inventories. It does NOT cover a
# failed inventory *download*: intersphinx logs that one without a warning type,
# and `suppress_warnings` matches on type only (`is_suppressed_warning` in
# sphinx.util.logging returns early when the type is None), so it cannot be
# silenced this way. That case is handled by the filter below instead.
suppress_warnings = ['intersphinx.external']

# -- Unreachable inventories -------------------------------------------------
# Under the project's strict ``-W`` build, an inventory that cannot be fetched
# is a hard failure -- so an offline build, a network blip, or one of the six
# upstream docs sites being briefly down breaks the build for a reason that has
# nothing to do with this project's sources. The message is therefore demoted to
# INFO: it still appears in the build log, but it no longer fails the build.
#
# Demoting that message is not enough on its own. Every `nx.Graph` or
# `pd.DataFrame` in a signature then resolves against nothing, and `nitpicky`
# reports each one: an offline build of this project produces 358 such
# warnings, none of which are real. So a build that lost an inventory also
# drops out of nitpicky mode -- the check cannot be performed honestly without
# the inventories it depends on, and running it anyway only reports their
# absence 358 times over.
#
# This is scoped to the degraded build alone. When all six inventories load --
# CI, Read the Docs, and any normal local build -- nitpicky stays fully on and
# a dead reference in prose still fails the build exactly as before.
import logging as _stdlib_logging

from sphinx.util import logging as _sphinx_logging

# Substring of the intersphinx message, stable across Sphinx 4-9. If a future
# release rewords it, the filter stops matching and the build fails loudly
# again -- the safe direction for this to break in.
_UNREACHABLE_INVENTORY = 'failed to reach any of the inventories'

# Set by the filter below, read once the inventories have been loaded.
_inventory_unreachable = False


class _DemoteUnreachableInventory(_stdlib_logging.Filter):
    """Log a failed inventory download as INFO rather than as a warning."""

    def filter(self, record):
        """Demote the download-failure record; pass everything else through."""
        global _inventory_unreachable
        if _UNREACHABLE_INVENTORY in str(record.msg):
            _inventory_unreachable = True
            record.levelno = _stdlib_logging.INFO
            record.levelname = 'INFO'
            # The status stream expects a trailing newline. Without it the next
            # progress line is appended to the end of the failure text.
            record.msg = f'{record.msg}\n'
        return True


def _relax_nitpicky_without_inventories(app):
    """Drop out of nitpicky mode if an inventory could not be downloaded.

    Returns: None
    """
    if not _inventory_unreachable:
        return
    app.config.nitpicky = False
    _sphinx_logging.getLogger('conf').info(
        'an intersphinx inventory could not be downloaded, so nitpicky mode is '
        'OFF for this build: every reference into an unreachable project would '
        'otherwise be reported as unresolved. Re-run with the inventories '
        'reachable to check cross-references.'
    )

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
html_css_files = ['accessibility.css']
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
    'titles_only': True,
}

# -- Protocol notebooks ------------------------------------------------------
# The protocol pages live as executed notebooks under examples/protocols/N/, so
# they render on GitHub and are committed with their outputs. Sphinx only reads
# documents below the source directory, so each notebook is copied to
# docs/source/examples/protocol_N.ipynb (git-ignored) before source discovery.
# This keeps the docnames `examples/protocol_N` unchanged, so the toctree in
# examples/index.rst and every {doc} cross-reference keep working.
import filecmp
import shutil

_CONF_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CONF_DIR, '..', '..'))

# Page docname -> notebook path relative to the repository root.
PROTOCOL_NOTEBOOKS = {
    f'examples/protocol_{n}': f'examples/protocols/{n}/protocol_{n}.ipynb'
    for n in (1, 2, 3, 4, 5)
}


def _copy_protocol_notebooks(app, config):
    """Copy the source notebooks into the docs tree before Sphinx reads them."""
    for docname, rel_path in PROTOCOL_NOTEBOOKS.items():
        src = os.path.join(_REPO_ROOT, rel_path)
        dst = os.path.join(app.srcdir, docname + '.ipynb')
        # copy2 preserves mtime, so unchanged notebooks don't trigger rebuilds.
        if not (os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)):
            shutil.copy2(src, dst)


def _edit_link_to_real_notebook(app, pagename, templatename, context, doctree):
    """Point 'Edit on GitHub' at the real notebook, not the build-time copy."""
    if pagename in PROTOCOL_NOTEBOOKS:
        meta = dict(context.get('meta') or {})
        meta['github_url'] = (
            f"https://github.com/{html_context['github_user']}/"
            f"{html_context['github_repo']}/blob/{html_context['github_version']}/"
            f"{PROTOCOL_NOTEBOOKS[pagename]}"
        )
        context['meta'] = meta


def setup(app):
    # Attached to intersphinx's own logger, not to the `sphinx` root: a stdlib
    # filter only sees records logged directly through the logger it is on, not
    # records propagated up from children. `getLogger` re-derives the real name
    # (Sphinx prefixes its own 'sphinx.') rather than hard-coding it here.
    _sphinx_logging.getLogger('sphinx.ext.intersphinx').logger.addFilter(
        _DemoteUnreachableInventory()
    )
    # intersphinx loads the inventories on `builder-inited` at the default
    # priority of 500, so a later priority sees the outcome of that load.
    app.connect('builder-inited', _relax_nitpicky_without_inventories,
                priority=900)
    app.connect('config-inited', _copy_protocol_notebooks)
    app.connect('html-page-context', _edit_link_to_real_notebook)
