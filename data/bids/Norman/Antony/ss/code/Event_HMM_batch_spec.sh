#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=Event_HMM_batch
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/Event_HMM_batch-%j.log

#module load anacondapy/5.1.0

for roitask in {1..1}
do
    sbatch Event_HMM_srun_spec.sh ${roitask}
done
