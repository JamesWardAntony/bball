#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -c 1
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --job-name=Event_decode_batch
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --output logs/Event_decode_batch-%j.log

#module load anacondapy/5.1.0

for cond in {1..10}
do
    sbatch Event_decode_srun.sh ${cond}
done
