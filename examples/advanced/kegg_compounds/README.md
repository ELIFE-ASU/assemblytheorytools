These scripts calculate assembly indices for KEGG compounds using CBRDB, a
curated biochemical database integrating KEGG and ATLAS data. They include a
local workflow for a list of KEGG compound IDs and an HPC job-array workflow
for distributing the calculations.

`kegg_c_complexity_matrix.py` computes the hydrogen-stripped assembly index
of KEGG compounds alongside several other molecular complexity scores
(Bertz, Böttcher, Wiener, Balaban, spacial score, Proudfoot and MC1), then
plots pairwise comparisons between the scores.

Run the local scripts from this directory. They use `wget` to fetch
`CBRdb_C.csv.zip` from CBRDB when it is absent, so either provide network access
and `wget` or place the archive at the path printed by the script.

The `job_array/` files are a Slurm template, not a portable submission script.
Before submitting, edit the `#SBATCH` array range, partition, QoS, time and
memory for your cluster, then provide the site-specific values as environment
variables:

```bash
export ATT_ASS_PATH=/path/to/asscpp
export ATT_DATA_DIR=/path/to/cbrdb-data
export ATT_ENV_NAME=ass_env
# Optional when the environment is not under $HOME/.conda/envs:
export ATT_PYTHON=/path/to/ass_env/bin/python
sbatch job_array/sub_sol_array.sh
```
