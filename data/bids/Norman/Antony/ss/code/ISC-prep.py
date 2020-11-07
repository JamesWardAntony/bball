#!/usr/bin/env python
# coding: utf-8

# In[29]:


ipynby=0 #python notebook or not
if ipynby==0:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject', type=str)
    args = parser.parse_args()
    print(args.subject)
    subject=args.subject
if ipynby==1:
    subject=1
if len(str(subject))==1:
    sub ='sub-0'+str(subject)
else:
    sub ='sub-'+str(subject)
subS = str(int(subject))
subs=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']#
ses = 'ses-01'
task='view'
n_trunc=3 # Number of volumes to trim/truncate
hrshiftval=4
neu350=0
print(sub)


# In[30]:


import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 
import glob
import time
from copy import deepcopy
import numpy as np
import pandas as pd 

from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
from nilearn.masking import intersect_masks
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import nibabel as nib
import scipy.io
from brainiak import image, io
if neu350==1:
    from isc_standalone import isc
else:
    from brainiak.isc import isc#, isfc, permutation_isc, timeshift_isc, phaseshift_isc
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.pyplot as plt 
if ipynby==1:
    get_ipython().run_line_magic('autosave', '5')
    get_ipython().run_line_magic('matplotlib', 'inline')
    sns.set(style = 'white', context='talk', font_scale=1, rc={"lines.linewidth": 2})


# In[31]:


from ss_utils import ss_dir, ss_bids_dir, ss_TR, ss_hrf_lag, run_names, n_runs
#mask_name=ss_bids_dir+'derivatives/freesurfer/fsaverage/mri/brainmask.mgz'
results_path=ss_bids_dir+'derivatives/secondlevel/'
firstlevel_dir=ss_bids_dir+'derivatives/firstlevel/'
firstlevel_sub_dir=ss_bids_dir+'derivatives/firstlevel/%s/' %(sub)
print(firstlevel_sub_dir)
analysis_dir=ss_dir+'analysis/'
mat_fname=analysis_dir+'d_event_mat.mat'
mat_contents = scipy.io.loadmat(mat_fname)
event_mat = mat_contents['event_mat'] #this matrix has the suspense bins for every TR
event_mat2 = mat_contents['event_mat2'] #this matrix has the suspense bins for every TR

orig_task_name = ['view']
ses='ses-01'
nS=20
tngs=9
dir_out = results_path + 'isc/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
    print('Dir %s created ' % dir_out)


# In[32]:


def get_file_names(sub_,data_dir_,n_trunc_,hrshiftval_, verbose = False):
    """
    Get all the participant file names
    
    Parameters
    ----------
    data_dir_ [str]: the data root dir
    task_name_ [str]: the name of the task 
    
    Return
    ----------
    fnames_ [list]: file names for all subjs
    """
    c_ = 0 
    fnames_ = []
    fname = os.path.join(
        data_dir_, '%s/%s_task-view_space-MNI152NLin2009cAsym_desc-preproc_bold_trim%s_%s_norm_event.nii.gz' % (sub_, sub_,n_trunc_,hrshiftval_))
    print(fname)
    # If the file exists
    if os.path.exists(fname):
        #print('Exists!')
        # Add to the list of file names 
        fnames_.append(fname)
        if verbose: 
            print(fname)
        c_+= 1
    return fnames_


# In[33]:


dv=2
inch=3
#find avg brain mask across subjects!
mask_imgs=[]
'''for subj in range(1,n_subjs_total+1):
    mask_fold = ss_bids_dir + 'derivatives/firstlevel/sub-0%s/masks/' % subj
    mask_name = mask_fold + 'sub-0%s_%s_brain.nii.gz' % (subj, ses)
    mask_imgs.append(mask_name)
    brain_mask1 = io.load_boolean_mask(mask_name)'''
for s in range(nS): 
    subj=subs[s]
    if len(str(subj))==1:
        subj1 ='sub-0'+str(subj)
    else:
        subj1 ='sub-'+str(subj)
    mask_fold = ss_bids_dir + 'derivatives/firstlevel/%s/masks/' % subj1
    mask_name = mask_fold + '%s_%s_brain.nii.gz' % (subj1, ses)
    mask_imgs.append(mask_name)
    brain_mask1 = io.load_boolean_mask(mask_name)


# In[34]:


# intersect 3 view brain masks    
avg_mask=intersect_masks(mask_imgs, threshold=0.5, connected=True)
#save x-sub avg mask
avg_mask_name = results_path + 'avg_brain_mask.nii.gz'
print(avg_mask_name)
dimsize=avg_mask.header.get_zooms()
affine_mat = avg_mask.affine
print(dimsize)
print(affine_mat)
hdr = avg_mask.header  # get a handle for the .nii file's header
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
nib.save(avg_mask, avg_mask_name)
print(avg_mask.shape)

#load in mask as boolean
brain_mask = io.load_boolean_mask(avg_mask_name)# Load the brain mask
coords = np.where(brain_mask)# Get the list of nonzero voxel coordinates
brain_nii = nib.load(avg_mask_name)# Load the brain nii image
print(brain_nii.shape)


# In[35]:


# load the functional data 
fnames = {}
images = {}
masked_images = {}
bold = {}
group_assignment = []
n_subjs = {}

for task_name in orig_task_name: 
    fnames[task_name] = get_file_names(sub,firstlevel_dir,n_trunc,hrshiftval)
    images[task_name] = io.load_images(fnames[task_name]) 
    #print(images[task_name])
   # images[task_name] = images[task_name].T #?? 
    masked_images[task_name] = image.mask_images(images[task_name], brain_mask) 
    #print(masked_images[task_name])
    # Concatenate all of the masked images across participants  
    bold[task_name] = image.MaskedMultiSubjectData.from_masked_images(
        masked_images[task_name], len(fnames[task_name]))
    if neu350==1:
        bold[task_name]=np.transpose(bold[task_name], [1,0,2])
        print(bold[task_name].shape)
    print(len(fnames[task_name]))
    # Convert nans into zeros
    bold[task_name][np.isnan(bold[task_name])] = 0
    print('Data loaded: {} \t shape: {}' .format(task_name, np.shape(bold[task_name])))
    bold_vol_event=[]
    bold_vol_event=np.zeros((avg_mask.shape[0], avg_mask.shape[1], avg_mask.shape[2], bold[task_name].shape[0]))
    bold_vol_event[coords[0], coords[1], coords[2], :] = bold[task_name].T
    print(bold_vol_event.shape)
    output_name = firstlevel_sub_dir + '%s_task-view_sb0_trim%d_%s.nii.gz' % (sub,n_trunc,hrshiftval)
    print(output_name)
    bold_nii = nib.Nifti1Image(bold_vol_event, affine_mat)
    hdr = bold_nii.header  # get a handle for the .nii file's header
    print(dimsize)
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2],1))
    nib.save(bold_nii, output_name)


# In[36]:


#split into suspense bins
bins=10
bincol=11
for bin in range(1,bins+1): #bins
    ind=event_mat[:,bincol]==bin
    print(ind)
    bold_temp=[]
    bold_temp=bold[task_name][ind,:]
    print(bold_temp.shape)
    bold_vol_event=[]
    bold_vol_event=np.zeros((avg_mask.shape[0], avg_mask.shape[1], avg_mask.shape[2], bold_temp.shape[0]))
    bold_vol_event[coords[0], coords[1], coords[2], :] = bold_temp.T
    print(bold_vol_event.shape)
    output_name = firstlevel_sub_dir + '%s_task-view_sb%s_trim%d_%s.nii.gz' % (sub,bin,n_trunc,hrshiftval)
    print(output_name)
    bold_nii = nib.Nifti1Image(bold_vol_event, affine_mat)
    hdr = bold_nii.header  # get a handle for the .nii file's header
    print(dimsize)
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2],1))
    nib.save(bold_nii, output_name)


# In[37]:


#split into game bins
bins=tngs
bincol=0
for bin in range(1,bins+1): #bins
    ind=event_mat[:,bincol]==bin
    print(ind)
    bold_temp=[]
    bold_temp=bold[task_name][ind,:]
    print(bold_temp.shape)
    bold_vol_event=[]
    bold_vol_event=np.zeros((avg_mask.shape[0], avg_mask.shape[1], avg_mask.shape[2], bold_temp.shape[0]))
    bold_vol_event[coords[0], coords[1], coords[2], :] = bold_temp.T
    print(bold_vol_event.shape)
    output_name = firstlevel_sub_dir + '%s_task-view_gb%s_trim%d_%s.nii.gz' % (sub,bin,n_trunc,hrshiftval)
    print(output_name)
    bold_nii = nib.Nifti1Image(bold_vol_event, affine_mat)
    hdr = bold_nii.header  # get a handle for the .nii file's header
    print(dimsize)
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2],1))
    nib.save(bold_nii, output_name)

