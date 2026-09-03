# Protocol 1: Calculating Assembly Indices

This protocol partially recapitulates Figure 1 from
[Sharma et al. (2023)](https://doi.org/10.1038/s41586-023-06600-9), which
introduced the molecular assembly index.

[`protocol_1.ipynb`](./protocol_1.ipynb) calculates the molecular assembly index
for a set of amino acids, both individually and as a joint system whose members
can share substructures, then applies the same idea to a plain character string.
It needs no external data. Every step is explained in the notebook, and the
heavy calculations are timed with `%%time`.

## Running it

Install JupyterLab (`pip install -e ".[notebooks]"` from the repository root, or
use the development conda environment), then open the notebook from this
directory so the figures it saves land beside it:

```bash
cd examples/protocols/1
jupyter lab protocol_1.ipynb
```

Re-run it headlessly to refresh the committed outputs:

```bash
cd examples/protocols/1
jupyter nbconvert --to notebook --execute --inplace protocol_1.ipynb
```

The notebook is committed with the outputs of a full run and is rendered in the
[documentation](https://assemblytheorytools.readthedocs.io/en/latest/examples/protocol_1.html).
