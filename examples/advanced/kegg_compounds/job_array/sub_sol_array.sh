#!/bin/bash
#SBATCH --job-name=J000
##SBATCH --output=output_%A_%a.out   # Output file for each array task
##SBATCH --error=error_%A_%a.out     # Error file for each array task
#SBATCH --array=0-18837              # Array of 10 jobs (adjust to your range) 4886963
#SBATCH --time=0-04:00:00            # Time limit
#SBATCH --ntasks=1                   # Number of tasks (1 per array task)
#SBATCH --cpus-per-task=1            # Number of CPU cores per taskó
#SBATCH --mem=4G                    # Memory per task 4 or 16
#SBATCH --partition=htc          # general lightwork highmem
#SBATCH --qos=public                 # private public

cd $SLURM_SUBMIT_DIR

: "${ATT_ASS_PATH:?Set ATT_ASS_PATH to your assemblyCPP executable}"
: "${ATT_DATA_DIR:?Set ATT_DATA_DIR to the directory containing CBRdb_C.csv.zip}"
: "${ATT_ENV_NAME:?Set ATT_ENV_NAME to your conda environment name}"

export ASS_PATH="$ATT_ASS_PATH"

module load mamba/latest
source activate "$ATT_ENV_NAME"

ATT_PYTHON="${ATT_PYTHON:-$HOME/.conda/envs/$ATT_ENV_NAME/bin/python3}"

# Run the Python script, passing the SLURM_ARRAY_TASK_ID to the script
"$ATT_PYTHON" calc_info.py "${SLURM_ARRAY_TASK_ID}" --data-dir "$ATT_DATA_DIR"
