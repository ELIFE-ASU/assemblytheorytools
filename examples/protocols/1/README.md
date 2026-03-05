# Protocol 1: Calculating Assembly Indices

This directory contains examples demonstrating the fundamental operations for calculating the Assembly Index (AI) for
different types of data. This is a partial recapitulation of Figure 1 from Sharma et al. (2023)
10.1038/s41586-023-06600-9

The first part of script (`protocol_1.py`) demonstrates how to calculate the Assembly Index for a set of molecules, both
individually and as a combined system. It performs the following steps:

1. **Input**: Defines a list of molecules by their names and SMILES strings.
2. **Individual Calculation**: Calculates the assembly index for each molecule separately in parallel.
3. **Joint Calculation**: Combines the molecules into a single system and calculates the "joint assembly index,"
   identifying shared substructures (virtual objects) in the process.
4. **Output**: Prints the individual and joint assembly indices, along with the SMILES strings of the virtual objects
   found in the combined pathway.
5. **Visualization**: Generates a molecular pathway plot for the combined system and saves it as
   `mol_pathway_example.svg` and `mol_pathway_example.png`.

The second part of script (`protocol_1.py`) extends the concept to arbitrary data sequences. It demonstrates how to:

1. **Input**: Takes a raw string input (e.g., `'gggfhhhvg'`).
2. **Calculation**: Calculates the Assembly Index, extracting the **virtual objects** (reused parts) found within the
   assembly pathway.
3. **Visualization**: Plots the assembly pathway for the string and saves the visualization as `str_pathway_example.svg`
   and `str_pathway_example.png`.

