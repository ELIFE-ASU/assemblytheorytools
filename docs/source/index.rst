assemblytheorytools documentation
=================================

A centralised set of tools for doing assembly theory calculations: computing
assembly indices for molecules, strings and arbitrary graphs, reconstructing and
enumerating assembly pathways, scoring molecular complexity, and plotting the
results.

Assembly theory quantifies the complexity of an object by the minimal number of
joining steps needed to build it from elementary parts, where every intermediate
that has already been made may be reused for free.
``assemblytheorytools`` (ATT) wraps three assembly calculators behind one Python
API: a C++ one for exact molecule, graph and string indices, a Rust one for fast
molecular indices, and a context-free-grammar one for approximate string
indices. The Rust and CFG calculators install alongside ATT as ordinary Python
dependencies; the C++ calculator is taken from ``ASS_PATH``, ``PATH`` or ATT's
cache, and built from source on first use only if none of those supplies one.

Installation
------------

ATT requires **Python 3.12 or newer**.

.. code-block:: bash

   python -m pip install assemblytheorytools

The first calculation below builds the C++ calculator if none is configured,
which takes a few minutes and needs ``git`` and a C++20 compiler. See
:doc:`install` for conda, HPC, Windows and build-from-source instructions.

Quick start
-----------

Compute the assembly index of caffeine:

.. code-block:: python

   import assemblytheorytools as att

   smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
   graph = att.smi_to_nx(smi)
   ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

   print(f"Assembly index: {ai}")
   print(f"Virtual objects: {len(virt_obj)}")

.. code-block:: text

   Assembly index: 9
   Virtual objects: 14

``ai`` is the assembly index. ``virt_obj`` holds the virtual objects — the
reusable intermediates found along the path — as graphs, which
:func:`~assemblytheorytools.tools_graph.nx_to_smi` converts back to SMILES.
``pathway`` is a :class:`~networkx.DiGraph` whose nodes are the virtual objects
and the joining steps, each carrying its object in a ``vo`` attribute.

Plot the pathway:

.. code-block:: python

   import matplotlib.pyplot as plt

   att.plot_pathway(pathway, plot_type="graph")
   plt.show()

Where to go next
----------------

* :doc:`route_map` — every quantity ATT computes, with its inputs, outputs
  and what it is used for.
* :doc:`concepts` — what the assembly index measures and how ATT represents it.
* :doc:`theory` — background on assembly theory: copy number, the assembly
  equation, the nested assembly spaces and what separates selectivity from
  selection.
* :doc:`glossary` — formal definitions of the assembly theory vocabulary.
* :doc:`guide/index` — task-oriented walkthroughs for molecules, strings,
  graphs, pathways, parallel runs, complexity scores and mass spectrometry.
* :doc:`examples/index` — the runnable scripts and the protocol notebooks
  shipped in the repository.
* :doc:`configuration` — environment variables and the calculator backends.
* :doc:`modules` — the full API reference.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   install
   route_map
   concepts
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Background
   :hidden:

   theory
   glossary

.. toctree::
   :maxdepth: 2
   :caption: Using the package
   :hidden:

   guide/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   modules

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   contributing
   citing
   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
