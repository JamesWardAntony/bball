#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=ImpConcat-Recall_batch
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/ImpConcat-Recall_sbatch-%j.log

#module load anacondapy/5.1.0
#module load pyger/0.9

for subject in {1..9}
do
    sbatch ImpConcat-Recall_srun.sh ${subject}
done
