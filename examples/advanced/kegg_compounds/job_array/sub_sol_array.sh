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

export ASS_PATH=/data/grp_swalke10/asscpp/v5_boost/asscpp_v5_boost_recursive

ENV_NAME="monsterproteinstability"
module load mamba/latest
source activate $ENV_NAME

# Run the Python script, passing the SLURM_ARRAY_TASK_ID to the script
$HOME/.conda/envs/$ENV_NAME/bin/python3 calc_info.py "${SLURM_ARRAY_TASK_ID}"
