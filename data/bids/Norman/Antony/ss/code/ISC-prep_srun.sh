#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=ISC-prep
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/ISC-prep_srun-%j.log

echo "subject: $1"

srun python -u ISC-prep.py -subject ${1}
