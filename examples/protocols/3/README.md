# Protocol 3: Joint Assembly Calculation

This script (`protocol_3.py`) demonstrates how to calculate the assembly index for a collection of molecules simultaneously. This technique is used to identify shared sub-structures and determine the "joint" complexity of a set of compounds, comparing it against their individual complexities.

It illustrates how to:
1.  **Data Preparation**: Convert common molecule names (e.g., "codeine", "diamorphine") to SMILES strings using the PubChem API and transform them into molecular graphs.
2.  **Individual vs. Joint Analysis**:
    *   Calculate **Individual Assembly Indices** for each molecule separately as a baseline.
    *   Combine the molecules into a single graph structure.
    *   Calculate the **Joint Assembly Index**, finding the most efficient pathway to construct *all* molecules in the set by leveraging shared "virtual objects" (sub-structures).
3.  **Visualization**:
    *   Extract and print the shared virtual objects found in the pathway.
    *   Plot the shared assembly pathway using various visualization styles, including standard cross-minimization and "metro map" layouts.