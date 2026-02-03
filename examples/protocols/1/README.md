# Protocol 1: Calculating Assembly Indices

This directory contains examples demonstrating the fundamental operations for calculating the Assembly Index (AI) for
different types of data. This is a partial recapitulation of Figure 1 from Sharma et al. (2023)
10.1038/s41586-023-06600-9

## Protocol 1a: Molecular Assembly Index

This script (`protocol_1a.py`) demonstrates how to calculate the Assembly Index for a chemical structure. It performs
the following steps:

1. **Input Conversion**: Converts a chemical name (e.g., 'diethyl phthalate') to a SMILES string via PubChem, and
   subsequently into a NetworkX graph.
2. **Calculation**: Computes the Molecular Assembly (MA) index, extracts virtual objects, and identifies the minimal
   construction pathway.
3. **Visualization**: Generates a molecular pathway plot using a 'crossmin_long' layout style to visualize the
   hierarchical assembly.
4. **Output**: Saves the visualization as `mol_pathway_example.svg` and `mol_pathway_example.png`.

## Protocol 1b: String Assembly Index

This script (`protocol_1b.py`) extends the concept to arbitrary data sequences. It demonstrates how to:

1. **Input**: Takes a raw string input (e.g., `'gggfhhhvg'`).
2. **Calculation**: Calculates the Assembly Index using `calculate_string_assembly_index`, extracting the **virtual
   objects** (reused parts) found within the assembly pathway.
3. **Visualization**: Plots the directed graph of the assembly pathway using a "metro" map layout style (
   `plot_digraph_metro`).