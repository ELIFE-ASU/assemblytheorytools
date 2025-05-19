---
title: 'Assembly Theory Tools: A Python Package for Molecular Assembly Pathway Analysis'
tags:
  - chemistry
  - cheminformatics
  - molecular assembly
  - graph theory
  - Python
authors:
  - name: Louie Slocombe
    orcid: TODO
    affiliation: 1
affiliations:
  - name: TODO (Your Institution)
    index: 1
date: YYYY-MM-DD
bibliography: paper.bib
---

# Summary

This package provides tools for the analysis and generation of molecular assembly pathways using graph-based methods. It enables the reconstruction, combination, and manipulation of molecular fragments and their assembly graphs, supporting cheminformatics research and automated molecule design.

Key features include:
- Construction and manipulation of directed graphs representing molecular assembly pathways.
- Combination of molecular fragments to generate new molecules.
- Layered graph operations for pathway analysis.
- Randomized construction of novel molecules from assembly pools.

# Statement of Need

Automated analysis and generation of molecular assembly pathways are essential for understanding chemical synthesis and exploring novel compound spaces. Existing tools often lack flexible graph-based manipulation and pathway reconstruction capabilities. This package addresses these needs by providing a Python-based framework for cheminformatics researchers.

# Implementation

The package is implemented in Python and leverages the NetworkX library for graph operations. It defines classes for molecules, assembly pools, and molecule spaces, supporting:
- Pathway reconstruction from SMILES strings.
- Graph-based combination and removal of molecular layers.
- Randomized generation of new molecules from existing assembly pools.

# Acknowledgements

# References

# Example

```python
import networkx as nx
from rdkit import Chem as Chem

import assemblytheorytools as att

if __name__ == "__main__":
    # Set the timeout duration for the assembly index calculation
    timeout = 600.0

    # List of SMILES strings representing the input molecules
    smiles_list = ['C(C(=O)O)N',
                   'C[C@@H](C(=O)O)N',
                   'C([C@@H](C(=O)O)N)O',
                   ]

    # Convert all SMILES strings to molecular graphs
    # Each SMILES string is converted to an RDKit molecule, then to a NetworkX graph,
    # and finally hydrogen atoms are removed from the graph
    graphs = [att.remove_hydrogen_from_graph(att.mol_to_nx(att.smi_to_mol(smile))) for smile in smiles_list]

    # Convert the list of molecular graphs into progressive union graphs (joint assembly)
    # This creates a series of disjoint union graphs, where each graph is the union of the
    # previous graph and the current one
    for i in range(1, len(graphs)):
        graphs[i] = nx.disjoint_union(graphs[i - 1], graphs[i])

    # Calculate the assembly index for each joint graph
    for i, graph in enumerate(graphs):
        print(f"Running joint: {i + 1}", flush=True)

        # Calculate the assembly index and virtual objects for the current graph
        ai, virt_obj, _ = att.calculate_assembly_index(graph,
                                                       # dir_code=dir_code,  # Optional directory for external code
                                                       timeout=timeout,
                                                       )

        # Flatten the dictionary of virtual objects into a list
        virt_obj = att.convert_pathway_dict_to_list(virt_obj)

        # Convert the virtual objects into SMILES strings
        smiles_output = [Chem.MolToSmiles(att.nx_to_mol(graph)) for graph in virt_obj]

        # Print the assembly index and the SMILES representation of the input graph
        print(f"Assembly index: {ai}", flush=True)
        print(f"Input graph: {smiles_output[0]}", flush=True)

        # Print the SMILES strings of the virtual objects
        print("VO SMILES:", flush=True)
        for smi in smiles_output[1:]:
            print(smi, flush=True)
        print(flush=True)
