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
and is not bundled with this repository.

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

### 1. Download the archive

Open the DOI above and use **Download Dataset** on the RADAR4Chem landing page
(47.5 MB, no account needed). You end up with a single archive named after the
DOI — `10.22000-OGoEQGlsZGElrgst.tar` — holding 4183 JCAMP-DX (`.jdx`) spectra
and one `meta_data.json`. Put it somewhere with room to spare beside it, since
step 3 unpacks it in place. Compression is autodetected, so a `.tar.xz` works
just as well.

### 2. Point the notebook at it

`DATASET` reads the `CHEMOTION_IR_ARCHIVE` environment variable and falls back
to a bare filename, so exporting the path is enough — no need to edit the
notebook:

```bash
export CHEMOTION_IR_ARCHIVE=~/Downloads/10.22000-OGoEQGlsZGElrgst.tar
```

### 3. What the processing cell does

There is no manual preparation step: the notebook's first timed cell calls
`att.process_chemotion_ir_data(DATASET)`, which unpacks and parses everything
for you. It

1. unpacks the archive into `chemotion_ir_data/` **beside the archive**, not
   beside the notebook;
2. reads `meta_data.json`, keeps the entries whose canonical SMILES RDKit can
   parse, and derives each one's spectrum `name` from its non-`.peak.jdx`
   identifier;
3. unpacks the nested `IR_data.tar.xz` into `IR_data/`, then loads every
   spectrum whose filename matches a surviving metadata name, in parallel across
   your cores;
4. merges the two on `name` and drops rows with any missing value.

What you get back is a DataFrame of three columns — `smiles`, `name` and
`spectrum` — where each spectrum is an `(N, 2)` array of wavenumber and
intensity. The cell then narrows it to molecules of at most `MAX_BONDS`
non-hydrogen bonds and drops spectra carrying non-finite intensities, which the
Savitzky-Golay smoother in the next step cannot process.

Afterwards the archive's directory looks like this, and the unpacked copies are
reused on later runs:

```text
10.22000-OGoEQGlsZGElrgst.tar
chemotion_ir_data/          <- created by step 1
├── meta_data.json
├── IR_data.tar.xz
└── IR_data/                <- created by step 3, holds the .jdx spectra
```

> **Leave `save=False`.** Passing `save=True` writes a
> `chemotion_ir_data.csv.gz` cache into your *working* directory, and the next
> call short-circuits to it — but CSV has no array type, so the reloaded
> `spectrum` column comes back as strings and the filtering step fails. If you
> have such a file lying about, delete it before re-running.

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

## Checking your run

The committed outputs come from a 28-core machine. Your counts should match even
though the timings will not:

| After | Molecules |
| --- | --- |
| loading, bond filter, finite-spectrum filter | 1267 (72 dropped as non-finite) |
| keeping spectra with 1–40 peaks | 1245 |
| exact assembly index | 1245 |

The linear fit then reports `r = 0.434`, `RMSD = 3.086`. The whole notebook took
22.6 s wall time, of which loading and processing the data was 12.6 s.
