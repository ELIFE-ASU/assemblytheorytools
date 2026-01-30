# Protocol 3: Joint Assembly Calculation

This script (`protocol_3.py`) demonstrates how to calculate the assembly index for a collection of molecules
simultaneously. This technique is used to identify shared sub-structures and determine the "joint" complexity of a set
of compounds, comparing it against their individual complexities. This is akin to Figure 6 of Liu et al. (2021)
10.1126/sciadv.abj2465

It illustrates how to:

1. **Pre-visualization**: Uses `show_common_bonds` to visually compare the input molecules (e.g., Codeine and Morphine)
   and highlight structural similarities before calculation.
2. **Individual vs. Joint Analysis**:
    * Calculates **Individual Assembly Indices** for each molecule separately as a baseline.
    * Combines the molecules into a single graph structure (`join_graphs`).
    * Calculates the **Joint Assembly Index**, finding the most efficient pathway to construct *all* molecules in the
      set by leveraging shared "virtual objects" (sub-structures).
3. **Pathway Visualization**:
    * Extracts and prints the shared virtual objects found in the pathway.
    * Plots the specific shared assembly pathway using the "metro map" layout (`plot_digraph_metro`), labeling nodes
      with synonyms where available.