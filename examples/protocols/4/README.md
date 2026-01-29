# Protocol 4: Correlating Assembly with IR Spectroscopy

This script (`protocol_4.py`) investigates the relationship between a physical observable—Infrared (IR) spectroscopy—and the theoretical Molecular Assembly Index. It demonstrates how to approximate assembly complexity using experimental data features. This is akin to Figure 3c of Jirasek et al. (2024) https://doi.org/10.1021/acscentsci.4c00120

It illustrates how to:
1.  **Data Processing & Filtering**: Load and process a Chemotion IR dataset, filtering molecules based on bond constraints and cleaning spectral data using Savitzky-Golay filters.
2.  **Feature Extraction**: Automatically identify and count spectral peaks within a specific range (e.g., 400-1500 cm⁻¹) using parallel processing.
3.  **Assembly Calculation**: Compute the Assembly Index for the corresponding molecules to establish ground truth complexity values.
4.  **Statistical Correlation**:
    *   Fit a linear model to estimate the Assembly Index based solely on the number of observed IR peaks.
    *   Evaluate the model using statistical metrics (Pearson correlation `r`, RMSD).
5.  **Visualization**:
    *   Generate heatmaps comparing **Assembly Index vs. Number of Peaks**.
    *   Plot **Observed vs. Predicted Assembly Index** to assess the predictive power of the spectral data.