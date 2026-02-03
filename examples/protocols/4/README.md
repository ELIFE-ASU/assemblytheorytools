# Protocol 4: Correlating Assembly with IR Spectroscopy

This script (`protocol_4.py`) investigates the relationship between a physical observable—Infrared (IR) spectroscopy—and
the theoretical Molecular Assembly Index. It demonstrates how to approximate assembly complexity using experimental data
features. This is akin to Figure 3c of Jirasek et al. (2024) https://doi.org/10.1021/acscentsci.4c00120

It illustrates how to:

1. **Data Processing & Filtering**: Loads and processes a Chemotion IR dataset, filters molecules based on bond
   constraints (max 30 NH bonds), and cleans spectral data using Savitzky-Golay filters.
2. **Molecule Visualization**: Generates visualizations for a sample molecule:
    * Plots the **IR Spectrum** with identified peaks.
    * Renders the **3D Atomic Structure** of the molecule.
3. **Feature Extraction**: Automatically counts spectral peaks for each molecule, filtering the dataset to include only
   those with 1 to 40 peaks.
4. **Assembly Calculation**: Computes the ground truth Assembly Index for the dataset using parallel processing.
5. **Statistical Correlation**:
    * Fits a linear model to estimate the Assembly Index based on the number of IR peaks.
    * Evaluates the model using Pearson correlation (`r`) and RMSD.
    * Generates a heatmap comparing **Observed vs. Predicted Assembly Index** to visualize the correlation.