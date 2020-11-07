#!/bin/bash

# Name of job?
#SBATCH --job-name=mriqc

# Where to output log files?
#SBATCH --output='../derivatives/mriqc/logs/mriqc-sub20-%A.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 3:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

subj=20 #update with SID 

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
# echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# # Set subject ID based on array index
# printf -v subj "%03d" $SLURM_ARRAY_TASK_ID

# PARTICIPANT LEVEL
echo "Running MRIQC on sub-$subj"

./run_mriqc.sh $subj

echo "Finished running MRIQC on sub-$subj"
date

# GROUP LEVEL
echo "Running MRIQC on group"

./run_mriqc_group.sh

echo "Finished running MRIQC on group"
date
