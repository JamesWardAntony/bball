#!/bin/bash

# Name of job?
#SBATCH --job-name=Import

# Where to output log files?
#SBATCH --output='./logs/extract_masked_data_ses01-%A.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 2:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=14000

#SBATCH --mail-user=jantony@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
# echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date


subjects='03'

for subject in $subjects
do
	echo "Importing sub-$subject"

	./run_notebook_01.sh $subject

	echo "Finished importing sub-$subject"
done

date
