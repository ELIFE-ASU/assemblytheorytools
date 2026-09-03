# Protocol 3: Correlating Assembly with IR Spectroscopy

This protocol follows the workflow behind Figure 3c of
[Jirasek et al. (2024)](https://doi.org/10.1021/acscentsci.4c00120).

[`protocol_3.ipynb`](./protocol_3.ipynb) correlates a physical observable, the
number of infrared (IR) peaks, with the molecular assembly index across a
dataset, and fits a linear model to estimate assembly index from IR peak count
alone. Every step is explained in the notebook, and the heavy steps are timed
with `%%time`.

## The data

The spectra come from the Chemotion IR collection, which is **external data**
and is not bundled with this repository. Download the archive from its DOI, then
either set the `CHEMOTION_IR_ARCHIVE` environment variable to its path or edit
the `DATASET` parameter in the notebook. The archive is named after its DOI, for
example `10.22000-OGoEQGlsZGElrgst.tar`, and is unpacked into a
`chemotion_ir_data/` directory beside itself.

> Jung, N., Tremouilhac, P., Punjabi, D., & Huang, P.-C. (2024). *Chemotion
> Repository - Data collection: FT-IR spectroscopy data (Chemotion IR)*
> [Data set]. Karlsruhe Institute of Technology.
> [doi:10.22000/OGoEQGlsZGElrgst](https://doi.org/10.22000/OGoEQGlsZGElrgst)

The collection is released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Cite it
alongside Jirasek et al. (2024) in anything you publish from this protocol, and
keep the attribution and licence on any spectra you redistribute or derive.
BibTeX for both is in [`att.bib`](../../../att.bib) (`jung2024chemotion`,
`jirasek2024molecular`).

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
