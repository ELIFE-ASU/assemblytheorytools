assemblytheorytools documentation
=================================

A centralised set of tools for doing assembly theory calculations: computing
assembly indices for molecules, strings and arbitrary graphs, reconstructing and
enumerating assembly pathways, scoring molecular complexity, and plotting the
results.

Installation
------------

.. code-block:: bash

   pip install assemblytheorytools

Quick start
-----------

.. code-block:: python

   from assemblytheorytools import calculate_assembly_index, smi_to_nx

   graph = smi_to_nx("CCO")
   assembly_index, virtual_objects, pathway = calculate_assembly_index(graph)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
