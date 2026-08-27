assemblytheorytools documentation
=================================

A centralised set of tools for doing assembly theory calculations: computing
assembly indices for molecules, strings and arbitrary graphs, reconstructing and
enumerating assembly pathways, scoring molecular complexity, and plotting the
results.

Assembly theory quantifies the complexity of an object by the minimal number of
joining steps needed to build it from elementary parts, reusing every
intermediate that has already been made. ``assemblytheorytools`` (ATT) wraps the
C++ and Rust assembly calculators behind one Python API. The C++ binaries ship
precompiled inside the package and the Rust calculator installs alongside it as
a wheel, so a calculation works out of the box.

Installation
------------

.. code-block:: bash

   pip install assemblytheorytools

The package bundles a precompiled assembly calculator, so nothing else is
required for the quick start below. See :doc:`install` for conda, HPC and
build-from-source instructions.

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
``pathway`` is a :class:`~networkx.DiGraph` in which each node is a virtual
object and each edge is a joining operation.

Plot the pathway:

.. code-block:: python

   import matplotlib.pyplot as plt

   att.plot_pathway(pathway, plot_type="graph")
   plt.show()

Where to go next
----------------

* :doc:`concepts` — what the assembly index measures and how ATT represents it.
* :doc:`theory` — background on assembly theory: copy number, the assembly
  equation, the nested assembly spaces and what separates selectivity from
  selection.
* :doc:`glossary` — formal definitions of the assembly theory vocabulary.
* :doc:`guide/index` — task-oriented walkthroughs for molecules, strings,
  graphs, pathways, parallel runs, complexity scores and mass spectrometry.
* :doc:`examples/index` — the runnable scripts and published protocols shipped
  in the repository.
* :doc:`configuration` — environment variables and the calculator backends.
* :doc:`modules` — the full API reference.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   install
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
