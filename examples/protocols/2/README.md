# Protocol 2: Large-Scale Molecular Assembly Analysis

This script (`protocol_2.py`) demonstrates an end-to-end workflow for analyzing the relationship between Molecular Weight (MW) and Molecular Assembly (MA) on a large scale.

It illustrates how to:
1.  **Data Acquisition**: Download and sample a large dataset (e.g., 20,000 molecules) from PubChem, filtering by criteria such as molecular weight limits and maximum bond counts.
2.  **Parallel Computation**: Efficiently convert raw SMILES strings to molecule objects and calculate their Assembly Indices using parallel processing tools (`att.mp_calc`, `att.calculate_assembly_parallel`) to handle computational load.
3.  **Visualization**:
    *   Plot **Molecular Weight vs. Assembly Index** as a standard scatter plot.
    *   Generate a **Heatmap** (density plot) to visualize the distribution of assembly values relative to molecular weight.