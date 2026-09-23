These scripts calculate assembly indices for KEGG compounds using CBRDB, a
curated biochemical database integrating KEGG and ATLAS data. They include three
local workflows and an HPC job-array workflow for distributing the
calculations.

`kegg_c_all.py` is the local workflow: it filters the CBRDB compound set down to
parseable, single-component molecules of at most 50 heavy atoms, calculates
their hydrogen-stripped assembly indices in parallel, writes
`kegg_c_assembly_index.csv`, and plots assembly index against heavy-atom count.

`kegg_c_compare.py` runs a smaller subset (at most 15 heavy atoms) and compares
assembly index against other complexity measures, producing heatmap, scatter and
three-dimensional plots.

`kegg_c_complexity_matrix.py` computes the hydrogen-stripped assembly index
of KEGG compounds alongside several other molecular complexity scores
(Bertz, Böttcher, Wiener, Balaban, spacial score, Proudfoot and MC1), then
plots pairwise comparisons between the scores.

`job_array/` holds the HPC workflow, described below.

Run the local scripts from this directory. They use `wget` to fetch
`CBRdb_C.csv.zip` from CBRDB when it is absent, so either provide network access
and `wget` or place the archive at the path printed by the script.

The `job_array/` files are a Slurm template, not a portable submission script.
Before submitting, edit the `#SBATCH` array range, partition, QoS, time and
memory for your cluster, then provide the site-specific values as environment
variables:

```bash
export ATT_ASS_PATH=/path/to/AssemblyCpp
export ATT_DATA_DIR=/path/to/cbrdb-data
export ATT_ENV_NAME=ass_env
# Optional when the environment is not under $HOME/.conda/envs:
export ATT_PYTHON=/path/to/ass_env/bin/python
sbatch job_array/sub_sol_array.sh
```
