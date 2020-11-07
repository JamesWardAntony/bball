#! /bin/bash

set -e #stop immediately if error occurs

subj=$1

data_dir=/jukebox/norman/jantony/surprisesuspense/data
raw_dir=$data_dir/conquest
extra_dir=$data_dir/extra
bids_dir=$data_dir/bids/Norman/Antony/ss
scripts_dir=$bids_dir/code

# STEP 5 deface T1 (run on local)
# cd /Volumes/norman/jantony/surprisesuspense/data/bids/Norman/Antony/ss
# code/deface.sh $1 01

# STEP 6 -- cleanup

# delete scout images
find $bids_dir/sub-$subj -name "*scout*" -delete

# rename fieldmaps to replace magnitude with epi (magnitude as part of filename is a remnant of dcm2niix)
mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_magnitude.json $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_epi.json
mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_magnitude.nii.gz $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_epi.nii.gz
mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_magnitude.json $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_epi.json
mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_magnitude.nii.gz $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_epi.nii.gz

# add "IntendedFor" to fieldmaps
# SESSION 1
beginning='"IntendedFor": ['
run1="\""ses-01/func/sub-${subj}_ses-01_task-view_run-01_bold.nii.gz"\","
run2="\""ses-01/func/sub-${subj}_ses-01_task-recall_run-01_bold.nii.gz"\","
run3="\""ses-01/func/sub-${subj}_ses-01_task-view_run-02_bold.nii.gz"\","
run4="\""ses-01/func/sub-${subj}_ses-01_task-recall_run-02_bold.nii.gz"\","
run5="\""ses-01/func/sub-${subj}_ses-01_task-view_run-03_bold.nii.gz"\","
run6="\""ses-01/func/sub-${subj}_ses-01_task-recall_run-03_bold.nii.gz"\""
end="],"

insert="${beginning}${run1} ${run2} ${run3} ${run4} ${run5} ${run6}${end}"

sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_epi.json
sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_epi.json

# # STEP 7 -- run BIDS validator
echo "Running bids validator"
/usr/people/rmasis/node_modules/.bin/bids-validator $bids_dir

# #To run bids validator in browser window
http://bids-standard.github.io/bids-validator/


