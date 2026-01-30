# Protocol 2: Large-Scale Molecular Assembly Analysis

This script (`protocol_2.py`) demonstrates an end-to-end workflow for analyzing the relationship between Molecular
Weight (MW) and Molecular Assembly (MA) on a large scale. This is akin to Figure 2b of Marshall et al. (2021)
10.1038/s41467-021-23258-x

It illustrates how to:

1. **Data Acquisition**: Samples a dataset (10,000 molecules) from the CBRDB database.
2. **Parallel Computation**: Efficiently converts raw SMILES strings to graph objects and calculates their Assembly
   Indices using parallel processing tools (`att.mp_calc`, `att.calculate_assembly_index_parallel`) to handle
   computational load.
3. **Visualization**:
    * Generates a **Heatmap** to visualize the density distribution of Assembly Index values relative to Molecular
      Weight.
    * Creates a **Molecule Grid** image displaying the structures and assembly indices for a subset of the analyzed
      molecules.