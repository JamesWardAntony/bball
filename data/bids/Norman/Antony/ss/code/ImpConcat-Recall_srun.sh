#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=ImpConcat-Recall_srun
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/ImpConcat-Recall_srun-%j.log

echo "subject: $1"

srun python -u ImpConcat.py -subject ${1}
