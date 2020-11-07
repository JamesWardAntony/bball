#!/bin/bash

# Run within BIDS code/ directory: sbatch slurm_fmriprep.sh

# Name of job?
#SBATCH --job-name=fmriprep

# Where to output log files?
#SBATCH --output='../derivatives/fmriprep/logs/fmriprep-sub13-%A.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 34:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=20000

#SBATCH --mail-user=myemail
#SBATCH --mail-type=BEGIN,END,FAIL

subj=13 #update with SID

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
# echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# Run fMRIPrep script with participant argument
echo "Running fMRIPrep on sub-$subj"

./run_fmriprep.sh $subj

echo "Finished running fMRIPrep on sub-$subj"
date
