#!/bin/bash
#SBATCH -t 00:10:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=GLM_srun
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/GLM_srun-%j.log

echo "subject: $1"

srun python -u GLM.py -subject ${1}
