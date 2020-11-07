#! /bin/bash

# STEP 1 copy dicoms from conquest to lab directory
# STEP 2 check that all volumes transferred
# STEP 3 unzip dicoms
# STEP 4 convert dicoms to BIDS

# GET CONQUEST DIRECTORY NAMES: 
# ls /jukebox/dicom/conquest/Skyra-AWP45031/Norman/2019

# RECOMMENDED TO RUN HEUDICONV IN TMUX WINDOW: 
# create a new tmux window: tmux new -s [name]
# attach to an existing window: tmux a -t [name]

# other scripts called by this one: number_of_files.py, run_heudiconv.py

# INPUTS: $1:subjectID   $2:session(00,01,02)   $3:conquest directory
# e.g. to run: ./step1_preproc.sh 101 00 101_0_SVD-0205-1651

module load anacondapy/5.3.1

subj=$1
session=$2
subj_dir=$3

skyra_dir=/jukebox/dicom/conquest/Prisma-MSTZ400D/Norman/2019 #this is where your files get sent to from the scanner
data_dir=/jukebox/norman/jantony/surprisesuspense/data #this is my study directory
raw_dir=$data_dir/conquest #this is where I want the data from conquest to be copied into my study directory 
extra_dir=$data_dir/extra #this is where defaced images or extra niftis will end up
bids_dir=$data_dir/bids/Norman/Antony/ss #this is where BIDS formatted data will end up and should match the program card on the scanner
scripts_dir=$bids_dir/code #directory with my preprocessing scripts, including this one

# reminder 
read -p "Are you running in a tmux window? Press any key if you want to continue" 

# STEP 1: copy dicoms from conquest to lab directory 
#echo "copying session $session dicoms"
#cp -r $skyra_dir/$subj_dir $raw_dir

# STEP 2: check that all volumes transferred
#echo "checking number of volumes transferred session $session"
#$scripts_dir/number_of_files.py $raw_dir/$subj_dir/dcm $raw_dir/check_volumes/ $subj $session
#output=${subj}_${session}_index.csv
# print output to terminal window
#cat $raw_dir/check_volumes/$output

# STEP 3: unzip dicoms
#echo "unzipping session $session dicoms"
#gunzip $raw_dir/$subj_dir/dcm/*.gz

# STEP 4: convert dicoms to BIDS
echo "converting to BIDS with heudiconv"
$scripts_dir/run_heudiconv.py $subj_dir $subj $session

# remove extra stuff generated by heudiconv
rm -rf $bids_dir/sourcedata/
rm -rf $bids_dir/.heudiconv/

