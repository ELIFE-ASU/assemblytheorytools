API reference
=============

``assemblytheorytools`` re-exports the commonly used functions of each submodule at
the package root, so ``from assemblytheorytools import calculate_assembly_index``
and ``from assemblytheorytools.assembly import calculate_assembly_index`` are
equivalent. The pages below document each submodule in turn.

.. note::

   The re-exported set is a curated subset, not every public name. The pages below
   document everything each submodule defines, so a name may appear here without
   being reachable as ``assemblytheorytools.<name>``. Lower-level helpers such as
   :func:`~assemblytheorytools.construction.tables_to_nx`,
   :class:`~assemblytheorytools.reassembler.ParsePathwayLog` and
   :func:`~assemblytheorytools.tools_atoms.calculate_goat` must be imported from
   the submodule that defines them::

      from assemblytheorytools.construction import tables_to_nx

   If ``att.<name>`` raises :exc:`AttributeError`, check the submodule page for the
   fully qualified import path.

Every module page opens with a summary table of the names it defines, followed by
the full documentation for each. If you are looking for a workflow rather than a
function, start from the :doc:`guide/index` instead.

Assembly index calculation
--------------------------

Computing the index itself, and everything derived from the pathway it produces.

.. toctree::
   :maxdepth: 1

   api/assembly
   api/recursive_ma
   api/construction
   api/find_other_paths
   api/neighborhood_enumeration
   api/reassembler

Molecule, graph and structure handling
--------------------------------------

Getting an object into the graph form the calculator expects, and back out again.

.. toctree::
   :maxdepth: 1

   api/tools_mol
   api/tools_graph
   api/tools_atoms
   api/tools_cell
   api/tools_string

Scoring and data
----------------

Alternative complexity measures, dataset access, and the spectroscopy pipelines.

.. toctree::
   :maxdepth: 1

   api/complexity_scores
   api/tools_data
   api/tools_mzml
   api/tools_ms_json

Utilities
---------

Plotting, file handling and parallel execution.

.. toctree::
   :maxdepth: 1

   api/tools_plotting
   api/tools_file
   api/tools_mp

Testing helpers
---------------

Small prebuilt graph fixtures (``water_graph``, ``co2_graph``, ``phosphine_graph``
and friends) used by the test suite. They are re-exported at the package root for
convenience when writing tests against ATT, but they are not analysis tools.

.. toctree::
   :maxdepth: 1

   api/tools_test

Package root
------------

.. automodule:: assemblytheorytools
   :no-members:
