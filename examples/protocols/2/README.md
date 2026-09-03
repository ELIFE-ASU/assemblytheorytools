# Protocol 2: Large-Scale Molecular Assembly Analysis

This protocol follows the workflow behind Figure 2b of
[Marshall et al. (2021)](https://doi.org/10.1038/s41467-021-23258-x).

[`protocol_2.ipynb`](./protocol_2.ipynb) samples 10,000 molecules from the CBRDB
database, computes their assembly indices in parallel, and visualises assembly
index against molecular weight as a density heatmap and a grid of example
structures. Every step is explained in the notebook, and the heavy steps are
timed with `%%time`.

This is the longest-running protocol. It downloads the CBRDB dataset over the
network on the first run and uses every CPU core. Lower `N_SAMPLES` in the
notebook for a quick look, and see the
[parallel calculations guide](https://assemblytheorytools.readthedocs.io/en/latest/guide/parallel.html)
for tuning the worker count to your machine.

## Running it

Install JupyterLab (`pip install -e ".[notebooks]"` from the repository root, or
use the development conda environment), then open the notebook from this
directory so the figures it saves land beside it:

```bash
cd examples/protocols/2
jupyter lab protocol_2.ipynb
```

Re-run it headlessly to refresh the committed outputs:

```bash
cd examples/protocols/2
jupyter nbconvert --to notebook --execute --inplace protocol_2.ipynb
```

The notebook is committed with the outputs of a full run and is rendered in the
[documentation](https://assemblytheorytools.readthedocs.io/en/latest/examples/protocol_2.html).
