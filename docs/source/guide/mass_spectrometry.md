# Spectroscopy and mass spectrometry

The assembly index of a molecule can be *estimated* from experimental spectra
without knowing the structure. This matters because it makes assembly index
measurable for unknown samples — the basis of using it as a biosignature.

Two independent routes are implemented: fragmentation trees from tandem mass
spectrometry, and peak counts from infrared spectra.

## Recursive MA from fragmentation trees

{mod}`assemblytheorytools.recursive_ma` estimates molecular assembly from an
MS/MS fragmentation tree. The tree is a nested dictionary keyed by *m/z*, each
value being the fragments that parent produced:

```python
import numpy as np
import assemblytheorytools as att

tree = {
    500.0: {
        400.0: {
            300.0: {200.0: {100.0: {}, 80.0: {}}, 150.0: {}},
            250.0: {150.0: {}, 100.0: {}},
        },
        350.0: {250.0: {}, 200.0: {100.0: {}}},
        150.0: {},
    }
}

att.rma_print_tree(tree)
print(att.rma_tree_depth(tree))
```

{class}`~assemblytheorytools.recursive_ma.MAEstimator` runs the estimate. It is
a Monte Carlo method, so it returns a distribution rather than a single number:

```python
estimator = att.MAEstimator(same_level=True, tol=0.5, n_samples=20, min_chunk=20.0)

estimate = estimator.estimate_ma(tree=tree, mw=400.0, progress_levels=0)

print(f"{np.mean(estimate):.2f} +/- {np.std(estimate):.2f}")
```

`tol` is the mass tolerance in Daltons — match it to the instrument, since too
tight a tolerance discards real fragments and too loose a one invents
relationships. `n_samples` trades runtime against the width of the estimate;
report the spread, not just the mean.

Module-level equivalents exist for the same steps
({func}`~assemblytheorytools.recursive_ma.rma_build_tree`,
{func}`~assemblytheorytools.recursive_ma.rma_estimate_ma`,
{func}`~assemblytheorytools.recursive_ma.rma_unify_trees`,
{func}`~assemblytheorytools.recursive_ma.rma_identify_parents`) if you would
rather not hold estimator state.

## Reading mzML files

{func}`~assemblytheorytools.tools_mzml.process_mzml_file` decodes an mzML
document — handling the compression schemes and binary precisions the format
allows — and writes the extracted spectra to a directory:

```python
att.process_mzml_file(
    filename="Sample.mzML",
    out_dir="mzml_output",
    rt_units="min",
    int_threshold=1000,
)
```

`int_threshold` drops peaks below an absolute intensity; set `relative=True` to
treat it as a fraction of the base peak instead.

{func}`~assemblytheorytools.tools_ms_json.process_mzml_json` consumes the JSON
representation produced from those files, which is the more convenient form once
the data has been extracted once.

{func}`~assemblytheorytools.tools_plotting.plot_ms2_spectrum` plots a processed
MS2 spectrum with its fragmentation tree overlaid.

[Protocol 4](../examples/protocol_4.md) runs this pipeline end to end on a real
stepped-MS3 sample.

## Infrared spectra

The second route counts peaks in an IR spectrum and fits a model against known
assembly indices.

```python
spectrum = att.load_ir_jcamp_data("sample.jdx")
spectrum = att.apply_sg_filter(spectrum, window_length=9, polyorder=3)

peaks = att.find_peak_indices_in_range(spectrum, min_x=400.0, max_x=1500.0)

att.plot_ir_spectrum(spectrum, peaks=peaks)
```

The 400–1500 cm⁻¹ window is the fingerprint region, where peak count tracks
structural complexity most closely.
{func}`~assemblytheorytools.tools_data.apply_sg_filter` applies a
Savitzky-Golay smooth first, because peak finding on raw spectra picks up noise;
{func}`~assemblytheorytools.tools_data.find_n_peak_indices_in_range` returns a
fixed number of the most prominent peaks when a consistent feature count is
needed across a dataset.

{func}`~assemblytheorytools.tools_data.process_chemotion_ir_data` loads and
cleans a whole Chemotion IR dataset into a DataFrame, and
{func}`~assemblytheorytools.tools_data.estimate_ai_from_ir_peaks` fits the model
that maps peak counts onto assembly index:

```python
params = att.estimate_ai_from_ir_peaks(peak_counts, ai_observed,
                                       att.linear_func, params_0=[1.0, 0.0])
```

{func}`~assemblytheorytools.tools_data.linear_func` through
{func}`~assemblytheorytools.tools_data.quintic_func` are the available model
forms, and {func}`~assemblytheorytools.tools_data.get_r`,
{func}`~assemblytheorytools.tools_data.get_r2` and
{func}`~assemblytheorytools.tools_data.get_rmsd` evaluate the fit. A linear
model is the published choice — prefer it unless a higher order is clearly
justified, since peak count is a coarse feature and a quintic will fit noise.

[Protocol 3](../examples/protocol_3.md) is the complete worked correlation.

## See also

* {doc}`../api/recursive_ma` — fragmentation trees and the MA estimator.
* {doc}`../api/tools_mzml` and {doc}`../api/tools_ms_json` — file parsing.
* {doc}`../api/tools_data` — spectral processing, fitting and statistics.
* {doc}`../examples/protocol_3`, {doc}`../examples/protocol_4` — worked pipelines.
