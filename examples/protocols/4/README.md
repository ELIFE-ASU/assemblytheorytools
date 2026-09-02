# Protocol 4: Estimating Assembly from Tandem Mass Spectrometry

This script (`protocol_4.py`) estimates the Molecular Assembly (MA) index of a compound directly from its tandem mass
spectrometry data, without using its structure. This is the measurement route
described by [Jirasek et al. (2024)](https://doi.org/10.1021/acscentsci.4c00120).

The test compound is a phosphonate ester, `COC(=O)C(NC(=O)OC(C)(C)C)P(=O)(OC)OC` (MW 297.2, parent m/z 296.26), whose
assembly index is independently known to be 14. The structure is used only to render the reference figure and to check
the final answer — the estimate itself sees nothing but the spectra.

It illustrates how to:

1. **Data Extraction**: Unpacks the bundled stepped-MS3 sample (`Sample_#15_Stepped_MS3.tar.xz`) and parses the mzML
   file with `att.process_mzml_file`, converting the result to structured DataFrames per MS level with
   `att.process_mzml_json`.
2. **Reference Visualization**: Renders the 3D atomic structure of the known compound with `att.plot_ase_atoms`.
3. **Spectral Filtering**: Reduces each MS level to its informative peaks using `att.rma_process`, applying a relative
   intensity floor (1%), a per-level absolute intensity floor, and a cap of 20 peaks per spectrum.
4. **Fragmentation Tree Construction**: Links parent and child ions within a 0.05 Da mass tolerance using
   `att.rma_identify_parents`, then assembles them into a tree with `att.rma_build_tree` and selects the parent closest
   to the target m/z.
5. **Visualization**: Plots the processed MS2 spectrum with its fragmentation tree overlaid via `att.plot_ms2_spectrum`,
   saved as `processed_MS2.svg` and `processed_MS2.png`.
6. **MA Estimation**: Compares two estimates from `att.MAEstimator` — a first approximation from molecular weight alone
   (`estimate_by_MW`), and the fragment-informed recursive estimate (`estimate_MA`) — against the known reference value
   of 14. The recursive estimate reports the shared sub-fragments it found, which are what let it improve on the
   mass-only guess.
