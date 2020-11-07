#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=E_HMM-ROI_batch
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/E_HMM-ROI_batch-%j.log

#module load anacondapy/5.1.0

for roitask in {1..27}
do
    sbatch Event_HMM-ROI_srun.sh ${roitask}
done
