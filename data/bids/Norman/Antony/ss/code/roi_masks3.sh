#! /bin/bash

module load fsl
module load freesurfer/6.0.0  

set -e #stop immediately when error occurs

subj=$1

SUBJ_DIR=sub-$subj
STUDY_DIR=/jukebox/norman/jantony/surprisesuspense
DATA_DIR=$STUDY_DIR/data
BIDS_DIR=$DATA_DIR/bids/Norman/Antony/ss
SCRIPT_DIR=$STUDY_DIR/code/
DERIV_DIR=$BIDS_DIR/derivatives
FIRSTLEVEL_DIR=$DERIV_DIR/firstlevel
FREESURFER_DIR=$DERIV_DIR/freesurfer/$SUBJ_DIR/mri
MASK_DIR=$FIRSTLEVEL_DIR/$SUBJ_DIR/masks
ROI_DIR=$BIDS_DIR/rois

MNI152NLin2009cAsym=$DERIV_DIR/fmriprep/$SUBJ_DIR/anat/${SUBJ_DIR}_desc-preproc_MNI152NLin2009cAsym.nii.gz
BOLD_REF=$DERIV_DIR/fmriprep/$SUBJ_DIR/ses-01/func/${SUBJ_DIR}_ses-01_task-view_run-01_space-MNI152NLin2009cAsym_boldref.nii.gz
aPARC=$FREESURFER_DIR/aparc.a2009s+aseg.mgz #parcellation is the cortical ribbon and segmentation are the subcortical volumes

# convert reference image to mgz
mri_convert $BOLD_REF $FIRSTLEVEL_DIR/$SUBJ_DIR/ses-01/${SUBJ_DIR}_ses-01_task-view_run-01_space-MNI152NLin2009cAsym_boldref.mgz

# convert aparc.aseg mgz label file into volume space (i.e., bold resolution...giving it boldref as the target space)
mri_label2vol --seg $aPARC --temp $BOLD_REF --o $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.mgz --fillthresh 0.5 --regheader $aPARC

# convert resample mgz label file to nifti 
mri_convert $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.mgz $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz
# mri_convert -rl $FIRSTLEVEL_DIR/$SUBJ_DIR/ses-00/${SUBJ_DIR}_ses-00_task-localizer_run-01_space-MNI152NLin2009cAsym_boldref.mgz -rt nearest $aPARC $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz

# now convert from Janice / Pieman
flirt -in $ROI_DIR/v1plus_3mm.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_V1.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_V1.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_V1.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_V1.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_V1.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_V1.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_V1.nii.gz

flirt -in $ROI_DIR/a1plus_2mm.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_A1.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_A1.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_A1.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_A1.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_A1.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_A1.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_A1.nii.gz

flirt -in $ROI_DIR/precun_fsl_thr50_3mm.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_precun.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_precun.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_precun.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_precun.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_precun.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_precun.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_precun.nii.gz

flirt -in $ROI_DIR/PMC_3mm.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz

flirt -in $ROI_DIR/PCC_2mm.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz

flirt -in $ROI_DIR/vmPFC_Pieman.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz

# from Janice
flirt -in $ROI_DIR/BHipp_3mm_thr30.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_HC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_HC.nii.gz -thr 30 -bin $MASK_DIR/${SUBJ_DIR}_HC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_HC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_HC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_HC.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_HC.nii.gz

# Neurosynth 
flirt -in $ROI_DIR/defaultmode_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -thr 7 -bin $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz

flirt -in $ROI_DIR/angulargyrus_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_ang.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_ang.nii.gz -thr 5 -bin $MASK_DIR/${SUBJ_DIR}_ang.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_ang.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_ang.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_ang.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_ang.nii.gz

flirt -in $ROI_DIR/nucleusaccumbens_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_NAcc.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_NAcc.nii.gz -thr 10 -bin $MASK_DIR/${SUBJ_DIR}_NAcc.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_NAcc.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_NAcc.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_NAcc.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_NAcc.nii.gz

flirt -in $ROI_DIR/ventralstriatum_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_VS.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_VS.nii.gz -thr 10 -bin $MASK_DIR/${SUBJ_DIR}_VS.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_VS.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_VS.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_VS.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_VS.nii.gz

flirt -in $ROI_DIR/dorsalstriatum_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_DS.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_DS.nii.gz -thr 7 -bin $MASK_DIR/${SUBJ_DIR}_DS.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_DS.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_DS.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_DS.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_DS.nii.gz

flirt -in $ROI_DIR/ventraltegmental_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_VTA.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_VTA.nii.gz -thr 11 -bin $MASK_DIR/${SUBJ_DIR}_VTA.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_VTA.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_VTA.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_VTA.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_VTA.nii.gz

flirt -in $ROI_DIR/amygdala_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_amyg.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_amyg.nii.gz -thr 11 -bin $MASK_DIR/${SUBJ_DIR}_amyg.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_amyg.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_amyg.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_amyg.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_amyg.nii.gz

flirt -in $ROI_DIR/anteriorcingulate_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_ACC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_ACC.nii.gz -thr 7 -bin $MASK_DIR/${SUBJ_DIR}_ACC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_ACC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_ACC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_ACC.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_ACC.nii.gz

flirt -in $ROI_DIR/ventromedialprefrontal_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_vmPFC2.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC2.nii.gz -thr 7 -bin $MASK_DIR/${SUBJ_DIR}_vmPFC2.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC2.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_vmPFC2.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC2.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_vmPFC2.nii.gz

flirt -in $ROI_DIR/orbitofrontalcortex_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_OFC.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_OFC.nii.gz -thr 5 -bin $MASK_DIR/${SUBJ_DIR}_OFC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_OFC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_OFC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_OFC.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_OFC.nii.gz

flirt -in $ROI_DIR/hippocampus_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_HC2.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_HC2.nii.gz -thr 5 -bin $MASK_DIR/${SUBJ_DIR}_HC2.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_HC2.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_HC2.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_HC2.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_HC2.nii.gz

#Murty probabilistic atlas
flirt -in $ROI_DIR/VTA_Murty.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_VTA2.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_VTA2.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_VTA2.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_VTA2.nii.gz -thr 0.75 -bin $MASK_DIR/${SUBJ_DIR}_VTA2.nii.gz

#McDougle RPE effect
flirt -in $ROI_DIR/zstatRPE_FSLMNI.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_RPE.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_RPE.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_RPE.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_RPE.nii.gz -thr 1 -bin $MASK_DIR/${SUBJ_DIR}_RPE.nii.gz

flirt -in $ROI_DIR/RPE_Cb_binary.nii -ref $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii -out $MASK_DIR/${SUBJ_DIR}_RPE_Cb.nii.gz -nosearch -applyxfm
fslmaths $MASK_DIR/${SUBJ_DIR}_RPE_Cb.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_ses-01_brain.nii $MASK_DIR/${SUBJ_DIR}_RPE_Cb.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_RPE_Cb.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_RPE_Cb.nii.gz

