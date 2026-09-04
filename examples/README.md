This folder contains worked examples for `assemblytheorytools`. Basic scripts
use packaged features directly; the protocol notebooks and advanced workflows
may require external datasets, optional dependencies, or an HPC environment.
Read the README beside an example before running it.

## Protocols

The `protocols` directory contains Jupyter notebooks that reproduce key findings
and workflows from published research, offering practical demonstrations of the
library's capabilities. Each is committed with its outputs. Install JupyterLab
with `pip install -e ".[notebooks]"` (or use the development conda environment)
and open one with `jupyter lab`; each protocol's README has the details.

### [Protocol 1: Calculating Assembly Indices](./protocols/1/)

Demonstrates the fundamental operations for calculating the Assembly Index (AI) for both molecules and strings. This
includes:

- Calculating the Molecular Assembly Index for individual molecules and combined systems.
- Calculating the String Assembly Index for arbitrary data sequences.
- Visualizing the resulting assembly pathways.

### [Protocol 2: Large-Scale Molecular Assembly Analysis](./protocols/2/)

Provides an end-to-end workflow for analyzing the relationship between Molecular Weight (MW) and Molecular Assembly (MA)
on a large scale. This includes:

- Acquiring and sampling data from a molecular database (CBRDB) and from PubChem.
- Performing large-scale, parallelized assembly calculations.
- Visualizing the results as heatmaps and a molecule grid, and comparing the two datasets.
- Running the same workflow on your own molecules, from a list or a CSV file, and placing them on the database heatmap.

### [Protocol 3: Correlating Assembly with IR Spectroscopy](./protocols/3/)

Investigates the relationship between a physical observable (Infrared spectroscopy) and the Molecular Assembly Index.
This includes:

- Processing and filtering experimental spectral data.
- Extracting features (e.g., peak counts) from spectra.
- Fitting a statistical model to predict the Assembly Index from spectral features and evaluating the correlation.

### [Protocol 4: Estimating Assembly from Tandem Mass Spectrometry](./protocols/4/)

Estimates the Molecular Assembly Index of a compound from its MS/MS spectra alone, without using its structure. This
includes:

- Parsing an mzML file and filtering the spectra at each MS level.
- Building a fragmentation tree by linking parent and child ions within a mass tolerance.
- Comparing a mass-only approximation against the fragment-informed recursive estimate, and both against a known
  reference value.

### [Protocol 5: Copy Number, Abundance and Ensemble Assembly](./protocols/5/)

Goes past the assembly index of a single object to the ensemble quantities that quantify selection, and to the question
of where the copy numbers they need actually come from. This includes:

- Composing a peptide's assembly index from the joint index of its amino acids and the assembly index of its sequence.
- Simulating peptide formation weighted by copy number, with and without a selection pressure, and measuring ensemble
  assembly and the exploration ratio on the result.
- Testing which way of turning mass-spectrometry abundance into copy numbers survives unknown ionisation efficiencies.
