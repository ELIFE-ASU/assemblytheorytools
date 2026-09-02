# Protocol 3: Correlating Assembly with IR Spectroscopy

This script (`protocol_3.py`) investigates the relationship between a physical observable—Infrared (IR) spectroscopy—and
the theoretical Molecular Assembly Index. It demonstrates how to approximate assembly complexity using experimental data
features. This follows the workflow behind Figure 3c of
[Jirasek et al. (2024)](https://doi.org/10.1021/acscentsci.4c00120).

The Chemotion IR archive used by the script is external data and is not bundled.
Pass its path on the command line.

It illustrates how to:

1. **Data Processing & Filtering**: Loads and processes a Chemotion IR dataset, keeps molecules with at most 30
   non-hydrogen bonds, and cleans spectral data using Savitzky-Golay filters.
2. **Single Molecule Visualization**: For a selected example molecule, it:
    * Plots the **IR Spectrum** with identified peaks.
    * Renders the **3D Atomic Structure**.
    * Calculates and prints its individual **Assembly Index**.
3. **Feature Extraction**: Automatically counts spectral peaks for each molecule in the entire dataset, filtering to
   include only those with 1 to 40 peaks.
4. **Large-Scale Assembly Calculation**: Computes the ground truth Assembly Index for the filtered dataset using
   parallel processing.
5. **Statistical Correlation**:
    * Fits a linear model to estimate the Assembly Index based on the number of IR peaks.
    * Evaluates the model using Pearson correlation (`r`) and RMSD.
    * Generates a heatmap comparing **Observed vs. Predicted Assembly Index** to visualize the correlation.
