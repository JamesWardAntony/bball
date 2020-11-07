#! /bin/bash

module load fsleyes

set -e #stop immediately when error occurs

subj=$1

SUBJ_DIR=sub-$subj
DATA_DIR=/jukebox/norman/jantony/surprisesuspense/data
BIDS_DIR=$DATA_DIR/bids/Norman/Antony/ss
DERIV_DIR=$BIDS_DIR/derivatives
ASHS_DIR=$DERIV_DIR/ashs
FIRSTLEVEL_DIR=$DERIV_DIR/firstlevel
OutputDir=$FIRSTLEVEL_DIR/$SUBJ_DIR/reg_t2_to_t1


T1=$DERIV_DIR/fmriprep/$SUBJ_DIR/anat/${SUBJ_DIR}_desc-preproc_T1w.nii.gz
T2=$BIDS_DIR/$SUBJ_DIR/ses-01/anat/${SUBJ_DIR}_ses-01_T2w.nii.gz

# CHECK ALIGNMENT
# # open merged T1w template, original T2, ASHS output, and aligned/resampled masks
# fsleyes $DERIV_DIR/fmriprep/$SUBJ_DIR/anat/${SUBJ_DIR}_desc-preproc_T1w.nii.gz $BIDS_DIR/$SUBJ_DIR/ses-01/anat/${SUBJ_DIR}_ses-01_T2w.nii.gz $ASHS_DIR/$SUBJ_DIR/final/*_corr_usegray.nii.gz $FIRSTLEVEL_DIR/$SUBJ_DIR/align_t2_to_t1.feat/${SUBJ_DIR}_left_corr_usegray_space-T1w.nii.gz $FIRSTLEVEL_DIR/$SUBJ_DIR/align_t2_to_t1.feat/${SUBJ_DIR}_right_corr_usegray_space-T1w.nii.gz &

# Option to open fsleyes and check everything:
# Load T1, aligned T2, and aligned/resampled left and right masks
fsleyes $T1 $OutputDir/${SUBJ_DIR}_T2w_to_T1w.nii.gz $OutputDir/${SUBJ_DIR}_left_corr_usegray_space-T1w.nii.gz $OutputDir/${SUBJ_DIR}_right_corr_usegray_space-T1w.nii.gz &