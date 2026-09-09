# Protocol 5: Copy Number, Abundance and Ensemble Assembly

This protocol follows
[Jirasek et al. (2026)](https://doi.org/10.48550/arXiv.2512.18752), which
quantifies the emergence of selection in prebiotic peptide chemistry.

[`protocol_5.ipynb`](./protocol_5.ipynb) is the one protocol here that goes
past the assembly index of a single object and into the ensemble quantities
built on top of it. It computes a peptide's assembly index as
`a_P = a_JAS + a_S`, simulates peptide formation under a copy-number-weighted
polymerisation model with and without an environmental selection pressure,
measures ensemble assembly `A` and the exploration ratio on the result, and
then asks the question a real measurement forces: mass spectrometry reports
intensity rather than copy number, so which way of turning abundance into
`n_i` actually recovers the answer? Every step is explained in the notebook,
and the heavy steps are timed with `%%time`.

## The data

There is none to fetch. The ensembles are simulated in the notebook, which
needs no network access and no external archive.

That is a deliberate substitution. The paper's experimental peptide ensembles
were annotated from LC-MS/MS and are not publicly available — its code
repository, `croningp/peptide_selection`, is not published — so the notebook
reimplements the phenomenological polymerisation model the paper describes
alongside them. Several details of that model are unstated in the paper and
are choices made here, which section 4 sets out. Read the results as a
reimplementation that reproduces the reported behaviour, not as the paper's
own numbers.

The simulation has one property the experiment does not: it knows the true
copy number of every sequence. Section 8 exploits that, deriving synthetic
mass-spectrometry intensities from the true counts so the different ways of
weighting `A` by abundance can be judged against a ground truth.

## Running it

Install JupyterLab (`pip install -e ".[notebooks]"` from the repository root,
or use the development conda environment), then open the notebook from this
directory so the figures it saves land beside it:

```bash
cd examples/protocols/5
jupyter lab protocol_5.ipynb
```

Re-run it headlessly to refresh the committed outputs:

```bash
cd examples/protocols/5
jupyter nbconvert --to notebook --execute --inplace protocol_5.ipynb
```

The notebook is committed with the outputs of a full run and is rendered in the
[documentation](https://assemblytheorytools.readthedocs.io/en/latest/examples/protocol_5.html).

## Checking your run

Everything below the joint assembly index of the amino acids is seeded, so
your numbers should match these exactly. Only the timings will differ; the
committed outputs come from a 28-core machine, though this protocol uses one
core and finishes in under ten seconds.

| Quantity | Expected |
| --- | --- |
| Joint assembly index of G, A, V, H and P | 12 |
| String assembly index of `GGGAPPHVPHVHHPVGGG` | 13 |
| Unique sequences across all six runs | 2245 |
| Pathways that are minimum | 2184 of 2245 (97.3%) |
| Exploration ratio, undirected | 0.894 ± 0.002 |
| Exploration ratio, directed | 0.685 ± 0.016 |
| Ensemble assembly, undirected | 3.50 ± 0.01 |
| Ensemble assembly, directed | 33.55 ± 2.29 |
| Variation in `A` between response draws at σ = 3 | 3% true, 7% equal, 90% intensity-weighted |

The two exploration ratios are the figures to compare against the paper, which
reports 0.85–0.95 for undirected polymerisation and 0.51–0.75 under proteases.
The absolute values of `A` are not comparable to any published number: they
depend on the copy-number convention, on the building blocks through
`a_JAS`, and on the scale of the simulation. Section 8 is about exactly that.
