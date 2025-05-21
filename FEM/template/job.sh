#!/bin/bash

#SBATCH --partition=p_pamish
#SBATCH --job-name="${SYSTEM}_${THETA}"
#SBATCH --output="${SYSTEM}_${THETA}_${JOB_NAME}.log"
#SBATCH --error="${SYSTEM}_${THETA}_${JOB_NAME}.error"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=72:00:00

if [[ -d "/scratch/pamish1/mian" ]]; then
    module purge
    module load gcc-9.2.0/openmpi/4.1.0
fi
source activate ML

python phase_field_model_FEM.py --system ${SYSTEM} --theta ${THETA} --job_name ${JOB_NAME} ${EXTRA_COMMANDS}
