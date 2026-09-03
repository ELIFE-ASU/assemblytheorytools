# Protocol 3: Correlating Assembly with IR Spectroscopy

This protocol follows the workflow behind Figure 3c of
[Jirasek et al. (2024)](https://doi.org/10.1021/acscentsci.4c00120).

[`protocol_3.ipynb`](./protocol_3.ipynb) correlates a physical observable, the
number of infrared (IR) peaks, with the molecular assembly index across a
dataset, and fits a linear model to estimate assembly index from IR peak count
alone. Every step is explained in the notebook, and the heavy steps are timed
with `%%time`.

The Chemotion IR archive is **external data** and is not bundled with the
repository. Download it, then either set the `CHEMOTION_IR_ARCHIVE` environment
variable to its path or edit the `DATASET` parameter in the notebook. The
archive is named after its DOI, for example `10.22000-OGoEQGlsZGElrgst.tar`. It
is unpacked into a `chemotion_ir_data/` directory beside the archive.

## Running it

Install JupyterLab (`pip install -e ".[notebooks]"` from the repository root, or
use the development conda environment), then open the notebook from this
directory so the figures it saves land beside it:

```bash
cd examples/protocols/3
CHEMOTION_IR_ARCHIVE=/path/to/10.22000-OGoEQGlsZGElrgst.tar jupyter lab protocol_3.ipynb
```

Re-run it headlessly to refresh the committed outputs:

```bash
cd examples/protocols/3
CHEMOTION_IR_ARCHIVE=/path/to/archive.tar \
    jupyter nbconvert --to notebook --execute --inplace protocol_3.ipynb
```

The notebook is committed with the outputs of a full run and is rendered in the
[documentation](https://assemblytheorytools.readthedocs.io/en/latest/examples/protocol_3.html).
