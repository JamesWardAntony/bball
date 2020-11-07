#! /bin/bash

# make sure pydeface is installed on your local machine (or have it installed on spock)
# SEE HERE: https://github.com/poldracklab/pydeface
# TO INSTALL:
# git clone https://github.com/poldracklab/pydeface.git
# cd pydeface
# python setup.py install

# from local terminal:
cd /jukebox/norman/jantony/surprisesuspense/data/bids/Norman/Antony/ss
# run with 2 inputs: subject ID and session ID (e.g., code/deface.sh 101 01 for subject 101, session 01)

sid=$1
session=$2

DATA_DIR=/jukebox/norman/jantony/surprisesuspense/data

# deface T1
T1=`find sub-$sid/ses-$session/anat -name "*T1w.nii.gz"`
pydeface $T1

# move defaced T1 to extra directory
T1_defaced=`find sub-$sid/ses-$session/anat -name "*T1w_defaced.nii.gz"`
mv $T1_defaced $DATA_DIR/extra/T1w_defaced
