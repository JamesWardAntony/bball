#!/usr/bin/env python

# make sure anacondapy/5.3.1 is loaded before running (otherwise run will not be imported from subprocess)

from glob import glob
from os.path import exists, join
from sys import argv
from subprocess import run

subj_dir = argv[1]
subj_id = argv[2]
session_id = argv[3]

fmri_dir = '/jukebox/norman/jantony/surprisesuspense/data'
raw_dir = join(fmri_dir, 'conquest') #folders copied from conquest
bids_dir = join(fmri_dir, 'bids', 'Norman', 'Antony', 'ss') 

print("Running heudiconv for subject {0} session {1}".format(subj_id, session_id))

run("singularity exec --cleanenv --bind {0}:/home "
    "/jukebox/hasson/singularity/heudiconv/heudiconv.sqsh "
    "heudiconv -f reproin --subject {2} --ses {3} --bids --locator /home/bids/Norman/Antony/ss --files "
    "/home/conquest/{1}".format(fmri_dir, subj_dir.split('/')[-1], subj_id, session_id),
    shell=True)
