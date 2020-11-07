#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=Event_HMM_srun
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/Event_HMM_srun-%j.log
#module load pyger/0.10.1
echo "roitask: $1"

srun python -u Event_HMM-ROI.py -roitask ${1}