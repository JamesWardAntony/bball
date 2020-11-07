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


# # open in Freesurfer (example)
# freeview -v derivatives/freesurfer/sub-04/mri/orig.mgz \
# derivatives/freesurfer/sub-04/mri/aparc.a2009s+aseg.mgz:colormap=lut:opacity=0.4 \
# -f derivatives/freesurfer/sub-04/surf/lh.white:annot=aparc.annot.a2009s

# Extract values for occipito-temporal mask - left, right, and bilateral (I used bilateral_oc-temp mask made here for localizer classification analysis)
# lookup table https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
# 400 V1 
# 26  Left-Accumbens-area, 58  Right-Accumbens-area
# 17  Left-Hippocampus, 53  Right-Hippocampus
# Orbital part of the inferior frontal gyrus (G_front_inf-Orbital label L:11113 R:12113 )
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 26 -uthr 26 -bin $MASK_DIR/${SUBJ_DIR}_lNAcc.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 58 -uthr 58 -bin $MASK_DIR/${SUBJ_DIR}_rNAcc.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_lNAcc.nii.gz -add $MASK_DIR/${SUBJ_DIR}_rNAcc.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_NAcc.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 17 -uthr 17 -bin $MASK_DIR/${SUBJ_DIR}_lHC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 53 -uthr 53 -bin $MASK_DIR/${SUBJ_DIR}_rHC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_lHC.nii.gz -add $MASK_DIR/${SUBJ_DIR}_rHC.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_HC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11113 -uthr 11113 -bin $MASK_DIR/${SUBJ_DIR}_left_frontal_inf-orbital.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12113 -uthr 12113 -bin $MASK_DIR/${SUBJ_DIR}_right_frontal_inf-orbital.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_frontal_inf-orbital.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_frontal_inf-orbital.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_frontal_inf-orbital.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 400 -uthr 400 -bin $MASK_DIR/${SUBJ_DIR}_V1.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 401 -uthr 401 -bin $MASK_DIR/${SUBJ_DIR}_V2.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_V1.nii.gz -add $MASK_DIR/${SUBJ_DIR}_V2.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_V1_V2.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 1180 -uthr 1180 -bin $MASK_DIR/${SUBJ_DIR}_l_ctx_STG.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 2180 -uthr 2180 -bin $MASK_DIR/${SUBJ_DIR}_r_ctx_STG.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 3180 -uthr 3180 -bin $MASK_DIR/${SUBJ_DIR}_l_wm_STG.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 4180 -uthr 4180 -bin $MASK_DIR/${SUBJ_DIR}_r_wm_STG.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_l_ctx_STG.nii.gz -add $MASK_DIR/${SUBJ_DIR}_r_ctx_STG.nii.gz -add $MASK_DIR/${SUBJ_DIR}_l_wm_STG.nii.gz -add $MASK_DIR/${SUBJ_DIR}_r_wm_STG.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_STG.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11121 -uthr 11121 -bin $MASK_DIR/${SUBJ_DIR}_left_oc-temp_lat-fusifor.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12121 -uthr 12121 -bin $MASK_DIR/${SUBJ_DIR}_right_oc-temp_lat-fusifor.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_oc-temp_lat-fusifor.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_oc-temp_lat-fusifor.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_lat-fusifor.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11123 -uthr 11123 -bin $MASK_DIR/${SUBJ_DIR}_left_oc-temp_med-Parahip.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12123 -uthr 12123 -bin $MASK_DIR/${SUBJ_DIR}_right_oc-temp_med-Parahip.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_oc-temp_med-Parahip.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_oc-temp_med-Parahip.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_med-Parahip.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11161 -uthr 11161 -bin $MASK_DIR/${SUBJ_DIR}_left_oc-temp_lat.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12161 -uthr 12161 -bin $MASK_DIR/${SUBJ_DIR}_right_oc-temp_lat.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_oc-temp_lat.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_oc-temp_lat.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_lat.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11162 -uthr 11162 -bin $MASK_DIR/${SUBJ_DIR}_left_oc-temp_med_and_Lingual.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12162 -uthr 12162 -bin $MASK_DIR/${SUBJ_DIR}_right_oc-temp_med_and_Lingual.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_oc-temp_med_and_Lingual.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_oc-temp_med_and_Lingual.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_med_and_Lingual.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_lat-fusifor.nii.gz -add $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_med-Parahip.nii.gz -add $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_lat.nii.gz -add $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp_med_and_Lingual.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_oc-temp.nii.gz
#visual
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11143 -uthr 11143 -bin $MASK_DIR/${SUBJ_DIR}_left_oc-pole.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12143 -uthr 12143 -bin $MASK_DIR/${SUBJ_DIR}_right_oc-pole.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11111 -uthr 11111 -bin $MASK_DIR/${SUBJ_DIR}_left_cun.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12111 -uthr 12111 -bin $MASK_DIR/${SUBJ_DIR}_right_cun.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_oc-pole.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_oc-pole.nii.gz -add $MASK_DIR/${SUBJ_DIR}_left_cun.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_cun.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_vc.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11133 -uthr 11133 -bin $MASK_DIR/${SUBJ_DIR}_left_temp_sup_gt.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11134 -uthr 11134 -bin $MASK_DIR/${SUBJ_DIR}_left_temp_sup_l.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11135 -uthr 11135 -bin $MASK_DIR/${SUBJ_DIR}_left_temp_sup_pp.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11136 -uthr 11136 -bin $MASK_DIR/${SUBJ_DIR}_left_temp_sup_pt.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12133 -uthr 12133 -bin $MASK_DIR/${SUBJ_DIR}_right_temp_sup_gt.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12134 -uthr 12134 -bin $MASK_DIR/${SUBJ_DIR}_right_temp_sup_l.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12135 -uthr 12135 -bin $MASK_DIR/${SUBJ_DIR}_right_temp_sup_pp.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12136 -uthr 12136 -bin $MASK_DIR/${SUBJ_DIR}_right_temp_sup_pt.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_left_temp_sup_gt.nii.gz -add $MASK_DIR/${SUBJ_DIR}_left_temp_sup_l.nii.gz -add $MASK_DIR/${SUBJ_DIR}_left_temp_sup_pp.nii.gz -add $MASK_DIR/${SUBJ_DIR}_left_temp_sup_pt.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_temp_sup_gt.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_temp_sup_l.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_temp_sup_pp.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_temp_sup_pt.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_ac.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11130 -uthr 11130 -bin $MASK_DIR/${SUBJ_DIR}_left_precun.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12130 -uthr 12130 -bin $MASK_DIR/${SUBJ_DIR}_right_precun.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_precun.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_precun.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_precun.nii.gz

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11125 -uthr 11125 -bin $MASK_DIR/${SUBJ_DIR}_left_ang.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12125 -uthr 12125 -bin $MASK_DIR/${SUBJ_DIR}_right_ang.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_ang.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_ang.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_ang.nii.gz

# now convert from Janice / Pieman rois
flirt -in $ROI_DIR/defaultmode_association-test.nii -ref $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -out $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz -thr 0.2 -bin $MASK_DIR/${SUBJ_DIR}_DMN.nii.gz

flirt -in $ROI_DIR/PMC_3mm.nii -ref $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -out $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PMC.nii.gz

flirt -in $ROI_DIR/PCC_2mm.nii -ref $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -out $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_PCC.nii.gz

flirt -in $ROI_DIR/vmPFC_Pieman.nii -ref $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -out $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -nosearch -applyxfm 
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -mul $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz -thr 0.5 -bin $MASK_DIR/${SUBJ_DIR}_vmPFC.nii.gz
