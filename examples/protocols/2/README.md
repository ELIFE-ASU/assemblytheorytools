# Protocol 2: Large-Scale Molecular Assembly Analysis

This protocol follows the workflow behind Figure 2b of
[Marshall et al. (2021)](https://doi.org/10.1038/s41467-021-23258-x).

[`protocol_2.ipynb`](./protocol_2.ipynb) samples 10,000 molecules from the CBRDB
database, computes their assembly indices in parallel, and visualises assembly
index against molecular weight as a density heatmap and a grid of example
structures. It then runs the same workflow on a random sample from PubChem
(`att.sample_random_pubchem`) and compares the two datasets on shared axes, and
ends with a worked example of running the workflow on your own molecules, from
a list or a CSV file, instead of a sampled database. Every step is explained in
the notebook, and the heavy steps are timed with `%%time`.

This is the longest-running protocol. It needs network access: it downloads the
CBRDB dataset on the first run and queries PubChem's PUG REST service for the
PubChem sample, which takes a few minutes at the default size. It also uses
every CPU core. Lower `N_SAMPLES` and `N_PUBCHEM` in the notebook for a quick
look, and see the
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
