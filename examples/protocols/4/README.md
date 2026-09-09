# Protocol 4: Estimating Assembly from Tandem Mass Spectrometry

This protocol implements the measurement route described by
[Jirasek et al. (2024)](https://doi.org/10.1021/acscentsci.4c00120).

[`protocol_4.ipynb`](./protocol_4.ipynb) estimates the molecular assembly (MA)
index of a compound directly from its tandem mass spectrometry data, without
using its structure. The test compound is a phosphonate ester,
`COC(=O)C(NC(=O)OC(C)(C)C)P(=O)(OC)OC` (MW 297.2, parent m/z 296.26), whose
assembly index is independently known to be 14. The structure is used only to
render the reference figure and to check the final answer; the estimate itself
sees nothing but the spectra. Every step is explained in the notebook, and the
heavy steps are timed with `%%time`.

The sample data is bundled as `Sample_#15_Stepped_MS3.tar.xz` and unpacked by the
notebook, so no download is needed. Running the notebook writes
`Sample_#15_Stepped_MS3.mzML` and a `temp_mzml_output/` directory beside it.

## Running it

Install JupyterLab (`pip install -e ".[notebooks]"` from the repository root, or
use the development conda environment), then open the notebook from this
directory so the figures it saves land beside it:

```bash
cd examples/protocols/4
jupyter lab protocol_4.ipynb
```

Re-run it headlessly to refresh the committed outputs:

```bash
cd examples/protocols/4
jupyter nbconvert --to notebook --execute --inplace protocol_4.ipynb
```

The notebook is committed with the outputs of a full run and is rendered in the
[documentation](https://assemblytheorytools.readthedocs.io/en/latest/examples/protocol_4.html).
