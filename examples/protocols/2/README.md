# Protocol 2: Large-Scale Molecular Assembly Analysis

This script (`protocol_2.py`) demonstrates an end-to-end workflow for analyzing the relationship between Molecular
Weight (MW) and Molecular Assembly (MA) on a large scale. This follows the
workflow behind Figure 2b of [Marshall et al. (2021)](https://doi.org/10.1038/s41467-021-23258-x).

It illustrates how to:

1. **Data Acquisition**: Samples a dataset (10,000 molecules) from the CBRDB database.
2. **Parallel Computation**: Efficiently converts raw SMILES strings to graph objects and calculates their Assembly
   Indices using parallel processing tools (`att.mp_calc`, `att.calculate_assembly_index_parallel`). This step includes
   a timeout, exact-mode failure handling, and hydrogen removal to manage computational load.
3. **Data Filtering**: Keeps exact results with an Assembly Index (AI) of 1 or greater, excluding failures and the
   valid but trivial AI 0 case from the plotted subset.
4. **Visualization**:
    * Generates a **Heatmap** to visualize the density distribution of Assembly Index values relative to Molecular
      Weight.
    * Creates a **Molecule Grid** image displaying the structures, nicknames, and assembly indices for a subset of the
      analyzed
      molecules, sorted by their assembly index.
