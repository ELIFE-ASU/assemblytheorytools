API reference
=============

``assemblytheorytools`` re-exports the public functions of every submodule at the
package root, so ``from assemblytheorytools import calculate_assembly_index`` and
``from assemblytheorytools.assembly import calculate_assembly_index`` are
equivalent. The pages below document each submodule in turn.

Assembly index calculation
--------------------------

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

.. toctree::
   :maxdepth: 1

   api/tools_mol
   api/tools_graph
   api/tools_atoms
   api/tools_cell
   api/tools_string

Scoring and data
----------------

.. toctree::
   :maxdepth: 1

   api/complexity_scores
   api/tools_data
   api/tools_mzml
   api/tools_ms_json

Utilities
---------

.. toctree::
   :maxdepth: 1

   api/tools_plotting
   api/tools_file
   api/tools_mp
   api/tools_test

Package root
------------

.. automodule:: assemblytheorytools
   :no-members:
