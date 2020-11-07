#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=Event_decode_srun
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/Event_decode_srun-%j.log
#module load pyger/0.9
echo "cond: $1"

srun python -u Event_decode.py -cond ${1}
