# Protocol 1: Calculating Assembly Indices

This directory contains examples demonstrating the fundamental operations for calculating the Assembly Index (AI) for different types of data.

## Protocol 1a: Molecular Assembly Index
This script demonstrates how to calculate the Assembly Index for a chemical structure. It typically takes a **SMILES** string as input, calculates the Molecular Assembly (MA) index, identifies the minimal construction pathway, and visualizes the hierarchical assembly tree of the molecule.

## Protocol 1b: String Assembly Index
This script (`protocol_1b.py`) extends the concept to arbitrary data sequences. It demonstrates how to:
1.  Calculate the Assembly Index for a raw string input (e.g., `'gggfhhhvg'`).
2.  Extract the **virtual objects** (reused parts) found within the assembly pathway.
3.  Generate and save visualizations of the assembly pathway (SVG/PNG) using a cross-minimization layout to display how the string is constructed from its basic units.