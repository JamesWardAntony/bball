#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=ISC-prep_batch
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/ISC-prep_batch-%j.log

#module load anacondapy/5.1.0
#module load pyger/0.9

for subject in {1..20}
do
    sbatch ISC-prep_srun.sh ${subject}
done
