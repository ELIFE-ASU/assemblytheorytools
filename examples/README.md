This folder contains various examples demonstrating how to use the features of the `assembly-theory-tools` project. Each
example is self-contained and includes instructions on how to run it. These examples cover a range of use cases and can
serve as a starting point for users looking to implement similar functionality in their own projects.

## Protocols

The `protocols` directory contains scripts that reproduce key findings and workflows from published research, offering
practical demonstrations of the library's capabilities.

### [Protocol 1: Calculating Assembly Indices](./protocols/1/)

Demonstrates the fundamental operations for calculating the Assembly Index (AI) for both molecules and strings. This
includes:

- Calculating the Molecular Assembly Index for individual molecules and combined systems.
- Calculating the String Assembly Index for arbitrary data sequences.
- Visualizing the resulting assembly pathways.

### [Protocol 2: Large-Scale Molecular Assembly Analysis](./protocols/2/)

Provides an end-to-end workflow for analyzing the relationship between Molecular Weight (MW) and Molecular Assembly (MA)
on a large scale. This includes:

- Acquiring and sampling data from a molecular database.
- Performing large-scale, parallelized assembly calculations.
- Visualizing the results as a heatmap and a molecule grid.

### [Protocol 3: Correlating Assembly with IR Spectroscopy](./protocols/3/)

Investigates the relationship between a physical observable (Infrared spectroscopy) and the Molecular Assembly Index.
This includes:

- Processing and filtering experimental spectral data.
- Extracting features (e.g., peak counts) from spectra.
- Fitting a statistical model to predict the Assembly Index from spectral features and evaluating the correlation.

