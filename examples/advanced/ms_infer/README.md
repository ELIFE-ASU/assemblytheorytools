This directory contains two mass-spectrometry workflows.

`ms_Infer.py` builds a fragmentation-tree JSON file from an external mzML file.
It requires the optional `pyopenms` library:

```bash
pip install pyopenms
python ms_Infer.py /path/to/sample.mzML
```

`ma_estimate_real_data.py` runs ATT's recursive estimator on the five bundled
pickle files under `recursive_ma/`; run it from this directory so those relative
paths resolve:

```bash
python ma_estimate_real_data.py
```

See [Jirasek et al. (2024), *Investigating and quantifying molecular complexity
using assembly theory and spectroscopy*](https://doi.org/10.1021/acscentsci.4c00120).
