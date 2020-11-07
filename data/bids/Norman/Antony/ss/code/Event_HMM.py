#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Script for running Hidden Markov models onviewing data
ipynby=0#python notebook or not
import numpy as np
import math
if ipynby==0:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-parceltask', type=str)
    args = parser.parse_args()
    print(args.parceltask)
    parceltask=args.parceltask
    parceltask=int(parceltask)
if ipynby==1:
    parceltask=96+47
subs=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']#
nS = len(subs)
ses='ses-01'
n_trunc=3 # Number of volumes to trim/truncate
hrshiftval=5
nP=96
boundtype = 0
neu350=1
if parceltask>(nP*nS):
    boundtype=1
    parceltask=parceltask-(nP*nS)
game_num=math.ceil(parceltask/nP)
all_task_names = ['view_gb%s' %game_num]
parcel_sel = (parceltask-1)%nP#parceltask
print(parcel_sel)
print(game_num)
print(all_task_names)
print(boundtype)


# In[3]:


import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 
import glob
import time
from copy import deepcopy

import pandas as pd 
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
from nilearn.masking import intersect_masks
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.datasets import load_mni152_template
import nibabel as nib
import scipy.io
from scipy import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve
import matplotlib.pyplot as plt
import seaborn as sns 
from numpy.polynomial.polynomial import polyfit
from brainiak import image, io

import brainiak.eventseg.event
from sklearn import decomposition
from sklearn.model_selection import LeaveOneOut, KFold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
if ipynby:
    get_ipython().run_line_magic('autosave', '5')
    get_ipython().run_line_magic('matplotlib', 'inline')
    sns.set(style = 'white', context='talk', font_scale=1, rc={"lines.linewidth": 2})
from ss_utils import ss_dir, ss_bids_dir, ss_TR, ss_hrf_lag, run_names, n_runs
from ss_utils import load_ss_epi_data, load_ss_mask, mask_data, load_data
results_path=ss_bids_dir+'derivatives/secondlevel/'
firstlevel_dir=ss_bids_dir+'derivatives/firstlevel/'
analysis_dir=ss_dir+'analysis/'
mat_fname=analysis_dir+'d_event_mat.mat'


# In[4]:


#find avg brain mask across subjects
mask_imgs=[]
for s in range(nS): 
    subj=subs[s]
    if len(str(subj))==1:
        sub ='sub-0'+str(subj)
    else:
        sub ='sub-'+str(subj)
    mask_fold = ss_bids_dir + 'derivatives/firstlevel/%s/masks/' % sub
    mask_name = mask_fold + '%s_%s_brain.nii.gz' % (sub, ses)    
    mask_imgs.append(mask_name)
    brain_mask1 = io.load_boolean_mask(mask_name)

# intersect 3 view brain masks    
#save x-sub avg mask
avg_mask_name = results_path + 'avg_brain_mask.nii.gz'
#load in mask as boolean
brain_mask = io.load_boolean_mask(avg_mask_name)# Load the brain mask
coords = np.where(brain_mask)# Get the list of nonzero voxel coordinates
brain_nii = nib.load(avg_mask_name)# Load the brain nii image
print(brain_nii.shape)


# In[5]:


def get_file_names(data_dir_, task_name_,n_trunc_,hrshiftval_, verbose = False):
    c_ = 0 
    fnames_ = []
    # Collect all file names 
    for s in range(nS): 
        subj=subs[s]
        if len(str(subj))==1:
            sub ='sub-0'+str(subj)
        else:
            sub ='sub-'+str(subj)
        #print(sub)
        fname = os.path.join(
            #data_dir_, 'sub-0%s/sub-0%s_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold_trim%s_norm_event.nii.gz' % (subj, subj, task_name_,n_trunc_))
            #data_dir_, '%s/%s_task-%s_trim%s.nii.gz' % (sub, sub, task_name_,n_trunc_))
            data_dir_, '%s/%s_task-%s_trim%d_%s.nii.gz' % (sub, sub, task_name_,n_trunc_,hrshiftval_))
        # If the file exists
        if os.path.exists(fname):
            print(sub)
            print('Exists!')
            # Add to the list of file names 
            fnames_.append(fname)
            if verbose: 
                print(fname)
            c_+= 1
    return fnames_

# load the functional data 
fnames = {}
images = {}
masked_images = {}
bold = {}
n_subjs = {}
for task_name in all_task_names: 
    fnames[task_name] = get_file_names(firstlevel_dir, task_name,n_trunc,hrshiftval)
    images[task_name] = io.load_images(fnames[task_name]) 
    #images[task_name] = images[task_name].T #?? 
    masked_images[task_name] = image.mask_images(images[task_name], brain_mask) 
    # Concatenate all of the masked images across participants  
    bold[task_name] = image.MaskedMultiSubjectData.from_masked_images(
        masked_images[task_name], len(fnames[task_name]))
    print(np.shape(bold[task_name]))
    # Convert nans into zeros
    bold[task_name][np.isnan(bold[task_name])] = 0
    # compute the group assignment label 
    n_subjs_this_task = np.shape(bold[task_name])[-1]
    n_subjs[task_name] = np.shape(bold[task_name])[-1]
    print('Data loaded: {} \t shape: {}' .format(task_name, np.shape(bold[task_name])))


# In[6]:


# parcels
from nilearn import image
from nilearn.image.image import mean_img
from nilearn.image import resample_to_img
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
i=np.eye(3)*3
atlas.maps=image.resample_img(atlas.maps, target_affine=i,interpolation='nearest') 
if ipynby==1:
    plotting.plot_roi(atlas.maps, title='the harvard-oxford parcel')
nP = len(atlas.labels)-1 # rm background region 
print('number of parcels:\t {}'.format(nP))
masker_ho = NiftiLabelsMasker(labels_img=atlas.maps)
index=masker_ho.labels_img.dataobj
parcels=atlas.labels
parcels[51]='Left Juxtapositional Lobule Cortex'
parcels[52]='Right Juxtapositional Lobule Cortex'
template = load_mni152_template()
template =image.resample_img(template, target_affine=i,interpolation='nearest')
print(parcels[62])


# In[7]:


# must first put  bold data into 3-D space for alignment with parcels
if neu350==1:
    if bold[task_name].shape[0]>10000: #if haven't already swapped dimensions
        bold[task_name]=np.transpose(bold[task_name], [1,0,2])
nTR =bold[task_name].shape[0]
#bold3d=np.zeros((nTR,brain_nii.shape[0],brain_nii.shape[1],brain_nii.shape[2],nS))
print(bold[task_name].shape)
curr3d = np.zeros(brain_nii.shape)
for parcel in range(parcel_sel,parcel_sel+1):#
    print(parcel)
    fill2 = index==parcel+1# rm background region 
    roi_coords=np.logical_and(brain_mask>0,fill2>0)
    for subj in range(nS):
        fill=np.zeros(brain_nii.shape)
        if subj == 0:
            nV =fill[roi_coords].shape[0]# Voxels in this parcel
            bold_parcel=np.zeros((nTR,nV,nS))
            #print(bold_parcel.shape)
        for TR in range(nTR):
            curr3d[coords]=bold[task_name][TR,:,subj]
            # now create a bold parcel data set for spatial ISC
            bold_parcel[TR,:,subj]=curr3d[roi_coords]
            
'''for subj in range(nS):
    for TR in range(nTR):
        current=bold[task_name][TR,:,subj]
        curr3d[coords]=current
        bold3d[TR,:brain_nii.shape[0],:brain_nii.shape[1],:brain_nii.shape[2],subj]=curr3d
for parcel in range(parcel_sel,parcel_sel+1):
    fill2 = index==parcel+1# rm background region 
    roi_coords=np.logical_and(brain_mask>0,fill2>0)
    bold_parcel=bold3d[:,roi_coords,:]'''
print(parcels[parcel_sel+1])


# In[8]:


from brainiak import image, io
mat_contents = scipy.io.loadmat(mat_fname)
event_mat = mat_contents['event_mat'] 
event_fl = mat_contents['event_fl'] #1st TR for event matrix
p_fl = mat_contents['p_fl'] #1st TR for possession
wc_fl = mat_contents['wc_fl'] #1st TR for possession
newposs = mat_contents['newposs']
smoothf=3
ses='ses-01'
if boundtype==0: #possessions
    posses=event_mat[:,0]==game_num #extract possessions from game
    maxedges=math.ceil(np.nanmax(event_mat[posses,3])) #max # possessions in this game
    fill=p_fl[:,0]==game_num
    true_edges=p_fl[fill,1]-event_fl[game_num-1,0] #find times of possession changes
    true_edges_sur=p_fl[fill,5] #find amount of surprise for each change in possession
elif boundtype==1: #Winning team flips
    wincontexts=event_mat[:,0]==game_num #extract win context changes from game
    maxedges=len(np.where(np.diff(event_mat[wincontexts,5]))[0])
    #math.ceil(np.nanmax(event_mat[wincontexts,5])) #max # changes in this game
    fill=wc_fl[:,0]==game_num
    true_edges=wc_fl[fill,1]-event_fl[game_num-1,0] #find times of changes
    true_edges_sur=wc_fl[fill,2] #find amount of surprise for each change
print(maxedges)
print(true_edges)
print(true_edges_sur) 

dir_out = results_path + 'decode/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
    print('Dir %s created ' % dir_out)


# In[9]:


#function for similarity matrix
def plot_tt_similarity_matrix_nobound(ax, data_matrix, n_TRs, title_text):
    ax.imshow(np.corrcoef(data_matrix.T), cmap='viridis')
    ax.set_title(title_text)
    ax.set_xlabel('TR')
    ax.set_ylabel('TR')
def plot_tt_similarity_matrix(ax, data_matrix, bounds, bounds_true, n_TRs, title_text):
    ax.imshow(np.corrcoef(data_matrix.T), cmap='viridis')
    ax.set_title(title_text)
    ax.set_xlabel('TR')
    ax.set_ylabel('TR')
    # plot the predicted boundaries 
    bounds_aug = np.concatenate(([0],bounds,[n_TRs]))
    for i in range(len(bounds_aug)-1):
        rect = patches.Rectangle(
            (bounds_aug[i],bounds_aug[i]),
            bounds_aug[i+1]-bounds_aug[i],
            bounds_aug[i+1]-bounds_aug[i],
            linewidth=2,edgecolor='w',facecolor='none')
        ax.add_patch(rect)
    bounds_aug = np.concatenate(([0],bounds_true,[n_TRs]))
    for i in range(len(bounds_aug)-1):
        rect = patches.Rectangle(
            (bounds_aug[i],bounds_aug[i]),
            bounds_aug[i+1]-bounds_aug[i],
            bounds_aug[i+1]-bounds_aug[i],
            linewidth=2,edgecolor='y',facecolor='none')
        ax.add_patch(rect)
#function for running_mean (added +1 on 9/26/19)
def running_max(x, N):
    rm=np.zeros((x.shape[0]))
    for i in range(len(x)):
        if i<N:
            rm[i]=np.max(x[0:i+N+1])#mean
        elif i>len(x)-1-N:
            rm[i]=np.max(x[i-N:len(x)-1+1])
        else:
            rm[i]=np.max(x[i-N:i+N+1])
    return rm
def running_mean(x, N):
    rm=np.zeros((x.shape[0]))
    for i in range(len(x)):
        if i<N:
            rm[i]=np.mean(x[0:i+N+1])#mean
        elif i>len(x)-1-N:
            rm[i]=np.mean(x[i-N:len(x)-1+1])
        else:
            rm[i]=np.mean(x[i-N:i+N+1])
    return rm
#simple violin plot
def vplot(true,dist,yl,rankd,sps,sp):
    tt='true percentile = %s' %rankd
    if sps>1:
        ax[sp].violinplot(dist, showextrema=False)
        ax[sp].scatter(1,true)
        ax[sp].xaxis.set_visible(False)
        ax[sp].set_ylabel(yl)
        ax[sp].set_title(tt)
    else:
        ax.violinplot(dist, showextrema=False)
        ax.scatter(1,true)
        ax.xaxis.set_visible(False)
        ax.set_ylabel(yl)
        ax.set_title(tt)


# In[10]:


D = np.transpose(bold_parcel, [1,0,2])
print(D.shape)
for subj in range(nS):
    zmat=np.std(D[:,:,subj],axis=1)>0
    if subj==0:
        zzmat=zmat
    else:
        zzmat=zzmat & zmat
D=bold_parcel[:,zzmat,0].T
V = D.shape[0] # number of voxels
#V = bold_parcel.shape[1] # number of voxels
K = maxedges # number of events
T = bold_parcel.shape[0] # Time points
bounds_subj_fixk=np.zeros((nS,T))
bounds_subj_smooth_fixk=np.zeros((nS,T))
print(V)
for subj in range(nS):
    D = bold_parcel[:,zzmat,subj].T
    zmat=D[:,1]!=0
    D=D[zmat,:]
    # Find the events in this dataset
    hmm_sim = brainiak.eventseg.event.EventSegment(K)
    hmm_sim.fit(D.T)
    pred_seg = hmm_sim.segments_[0]
    # extract the boundaries 
    bs=np.where(np.diff(np.argmax(pred_seg, axis=1)))[0]
    bounds_subj_fixk[subj,bs] = 1 # mark the boundaries in the continuous space
    bounds_subj_smooth_fixk[subj,:]=running_mean(bounds_subj_fixk[subj,:],smoothf)
    if ipynby==1:
        if subj==0: # plot the data for sample subject
            f, ax = plt.subplots(1,1, figsize=(6, 2))
            ax.imshow(D, interpolation='nearest', cmap='viridis', aspect='auto')
            ax.set_ylabel('Voxels')
            ax.set_title('Brain activity for sample subject')
            ax.set_xlabel('TRs')

            f, ax = plt.subplots(1,1, figsize = (5,4))
            ax.imshow(np.corrcoef(D.T), cmap='viridis')
            title_text = 'TR-TR correlation matrix'
            ax.set_title(title_text)
            ax.set_xlabel('TR')
            ax.set_ylabel('TR')

            f, ax = plt.subplots(1,1, figsize=(8, 4))
            ax.imshow(hmm_sim.event_pat_.T, cmap='viridis', aspect='auto')
            ax.set_title('Estimated brain pattern for each event')
            ax.set_ylabel('Event ID')
            ax.set_xlabel('Voxels')

            f, ax = plt.subplots(1,1,figsize=(12,4))
            this_img=pred_seg.T
            ax.imshow(this_img, aspect='auto', cmap='viridis', interpolation='none')
            ax.set_xlabel('Timepoints')
            ax.set_ylabel('Event label')
            ax.vlines(true_edges,0,this_img.shape[0])
            ax.set_title('Predicted event segmentation, by HMM with the ground truth n_events')
            f.tight_layout()
            f, ax = plt.subplots(1,1, figsize = (10,8))
            title_text = 'TR-TR correlation matrix'
            plot_tt_similarity_matrix_nobound(ax, D, T, title_text)
            f.tight_layout()
            fign=ss_dir+'pics/HMMFixk-PredSeg-OneSubNoBound-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
            f.savefig(fign)
            f, ax = plt.subplots(1,1, figsize = (10,8))
            title_text = '''
            Overlay the predicted event boundaries
            and true event boundaries
            on top of the TR-TR correlation matrix
            '''
            plot_tt_similarity_matrix(ax, D, bs, true_edges, T, title_text)
            f.tight_layout()
            fign=ss_dir+'pics/HMMFixk-PredSeg-OneSub-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
            f.savefig(fign)


# In[15]:


#find boundaries / agreement using a fixed K, With and without smoothing
sum_bounds_fixk=np.mean(bounds_subj_fixk,axis=0)
sum_bounds_smooth_fixk=running_mean(sum_bounds_fixk,smoothf)
true_edges_cont=np.zeros((T))
true_edges_cont[true_edges[1:].astype(int)]=1
true_edges_cont_smooth=running_max(true_edges_cont,smoothf)
#Compute correlation between possession boundaries and HMM boundaries
fill=np.corrcoef(sum_bounds_fixk,true_edges_cont)
agree_fixk_r=fill[1,0]
fill=np.corrcoef(sum_bounds_smooth_fixk,true_edges_cont_smooth)
smooth_agree_fixk_r=fill[1,0]
if ipynby==1:
    f, ax = plt.subplots(2,1, figsize = (14,10))
    ax[0].plot(sum_bounds_fixk)
    ax[0].set_ylabel('Agreement across subjects')
    ax[0].set_title('Correlation = %s' %agree_fixk_r)
    ax[0].plot(np.max(sum_bounds_fixk)*(true_edges_cont/np.max(true_edges_cont)))
    ax[1].plot(sum_bounds_smooth_fixk)
    ax[1].set_ylabel('Agreement across subjects')
    ax[1].set_title('Correlation = %s' %smooth_agree_fixk_r)
    ax[1].set_xlabel('TRs')
    ax[1].plot(np.max(sum_bounds_smooth_fixk)*(true_edges_cont_smooth/np.max(true_edges_cont_smooth)))
    f.tight_layout()
    fign=ss_dir+'pics/HMMagreementCorrs-fixk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[16]:


#run correlation against amount of surprise rather than binary possession change (1/0)
true_edges_cont_sur=np.zeros((T))
true_edges_cont_sur[true_edges[1:].astype(int)]=true_edges_sur[1:]
true_edges_cont_sur_smooth=running_max(true_edges_cont_sur,smoothf)
ind=true_edges_cont_sur>0
fill=np.corrcoef(sum_bounds_fixk[ind],true_edges_cont_sur[ind])
agree_fixk_sur_r=fill[1,0]
ind_s=true_edges_cont_sur_smooth>0
fill=np.corrcoef(sum_bounds_smooth_fixk[ind_s],true_edges_cont_sur_smooth[ind_s])
smooth_agree_fixk_sur_r=fill[1,0]
if ipynby==1:
    f, ax = plt.subplots(2,1, figsize = (14,10)) 
    ax[0].plot(sum_bounds_fixk[ind])
    ax[0].set_ylabel('''Agreement across subjects, 
    surprise, fix k''')
    ax[0].set_title('Correlation = %s' %agree_fixk_sur_r)
    ax[0].plot(np.max(sum_bounds_fixk[ind])*(true_edges_cont_sur[ind]/np.max(true_edges_cont_sur[ind])))
    ax[1].plot(sum_bounds_smooth_fixk[ind_s])
    ax[1].set_ylabel('''Agreement across subjects, 
    surprise, fixed k, smooth''')
    ax[1].set_title('Correlation = %s' %smooth_agree_fixk_sur_r)
    ax[1].set_xlabel('>0 surprise time points')
    ax[1].plot(np.max(sum_bounds_smooth_fixk[ind_s])*(true_edges_cont_sur_smooth[ind_s]/np.max(true_edges_cont_sur_smooth[ind_s])))
    f.tight_layout()
    fign=ss_dir+'pics/HMMagreementCorrs-Sur-fixk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[62]:


#test against null distribution of scrambled boundaries, all using a fixed K
iters=1000
true_edges_edge=np.hstack([true_edges,T]) #create all the way up to the actual last TR
print(true_edges_edge)
poss_sizes=np.diff(true_edges_edge) #find the length of each possession for scrambling
print(poss_sizes)
agree_fixk_iter_r=np.zeros((iters))
smooth_agree_fixk_iter_r=np.zeros((iters))
agree_fixk_iter_sur_r=np.zeros((iters))
smooth_agree_fixk_iter_sur_r=np.zeros((iters))
jj=0
#for iter in range(0,iters):
while jj<1000:
    rps=np.random.permutation(poss_sizes.shape[0])
    true_edges_rp=np.cumsum(poss_sizes[rps]) #assign new edges
    true_edges_rp=np.hstack([0,true_edges_rp[:-1]]) #add 0 at beginning and cut last row to make similar to true_edges
    true_edges_rp_cont=np.zeros((T))
    true_edges_rp_cont[true_edges_rp[1:].astype(int)]=1 #put in continuous space
    true_edges_rp_cont_smooth=running_max(true_edges_rp_cont,smoothf)
    pss=np.random.permutation(true_edges_sur.shape[0]-1)
    pss=pss+1 #Make sure it doesn't select the first event
    true_edges_rp_cont_sur=np.zeros((T))
    true_edges_rp_cont_sur[true_edges_rp[1:].astype(int)]=true_edges_sur[pss]#
    true_edges_rp_cont_sur_smooth=running_max(true_edges_rp_cont_sur,smoothf)
    #find correlations between scrambled boundaries and those found by HMM
    fill=np.corrcoef(sum_bounds_fixk,true_edges_rp_cont)
    agree_fixk_iter_r[jj]=fill[1,0]
    fill=np.corrcoef(sum_bounds_smooth_fixk,true_edges_rp_cont_smooth)
    smooth_agree_fixk_iter_r[jj]=fill[1,0]
    ind=true_edges_rp_cont_sur>0
    fill=np.corrcoef(sum_bounds_fixk[ind],true_edges_rp_cont_sur[ind])
    agree_fixk_iter_sur_r[jj]=fill[1,0]
    ind_s=true_edges_rp_cont_sur_smooth>0
    fill=np.corrcoef(sum_bounds_smooth_fixk[ind_s],true_edges_rp_cont_sur_smooth[ind_s])
    smooth_agree_fixk_iter_sur_r[jj]=fill[1,0]
    if ~np.isnan(agree_fixk_iter_sur_r[jj]):
        jj=jj+1


# In[63]:


agree_fixk_rankd=stats.percentileofscore(agree_fixk_iter_r,agree_fixk_r)
smooth_agree_fixk_rankd=stats.percentileofscore(smooth_agree_fixk_iter_r,smooth_agree_fixk_r)
agree_fixk_sur_rankd=stats.percentileofscore(agree_fixk_iter_sur_r,agree_fixk_sur_r)
smooth_agree_fixk_sur_rankd=stats.percentileofscore(smooth_agree_fixk_iter_sur_r,smooth_agree_fixk_sur_r)
if ipynby==1: #plot true versus null distribution, all using a fixed K
    f, ax = plt.subplots(2,1, figsize = (5,10))
    yl='''True vs %s scrambled event 
    boundaries, fixed k, no smoothing''' %iters
    vplot(agree_fixk_r,agree_fixk_iter_r,yl,agree_fixk_rankd,2,0)
    yl='''True vs %s scrambled event 
    boundaries, fixed k, smoothing''' %iters
    vplot(smooth_agree_fixk_r,smooth_agree_fixk_iter_r,yl,smooth_agree_fixk_rankd,2,1)
    f.tight_layout()
    fign=ss_dir+'pics/HMMFixk-AgreementVsIters-bestk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)
    #same for surprise
    f, ax = plt.subplots(2,1, figsize = (5,10))
    yl='''True vs %s scrambled event 
    boundaries, surprise, 
    fixed k, no smoothing''' %iters
    vplot(agree_fixk_sur_r,agree_fixk_iter_sur_r,yl,agree_fixk_sur_rankd,2,0)
    yl='''True vs %s scrambled event 
    boundaries, surprise, 
    fixed k, smoothing''' %iters
    vplot(smooth_agree_fixk_sur_r,smooth_agree_fixk_iter_sur_r,yl,smooth_agree_fixk_sur_rankd,2,1)
    f.tight_layout()
    fign=ss_dir+'pics/HMMFixk-SurAgreementVsIters-bestk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)
    


# In[64]:


# test accuracy at the boundaries by testing within and across boundary correlations
# fit HMM with fixed K as the number of possessions
k = maxedges
w = 5  # window size
nPerm = 1000
D = np.transpose(bold_parcel, [1,0,2])
print(D.shape)
D=D[zzmat,:,:]
    
nTR = bold_parcel.shape[0]
print(D.shape)

within_across_fixk = np.zeros((nS, nPerm+1))
for left_out in range(nS):
    # Fit to all but one subject
    ev = brainiak.eventseg.event.EventSegment(k)
    ev.fit(D[:,:,np.arange(nS) != left_out].mean(2).T)
    events = np.argmax(ev.segments_[0], axis=1)

    corrs = np.zeros(nTR-w) # Compute correlations separated by w in time
    for t in range(nTR-w):
        corrs[t] = pearsonr(D[:,t,left_out],D[:,t+w,left_out])[0]
    _, event_lengths = np.unique(events, return_counts=True)

    # Compute within vs across boundary correlations, for real and permuted bounds
    np.random.seed(0)
    for p in range(nPerm+1):
        within = corrs[events[:-w] == events[w:]].mean()
        across = corrs[events[:-w] != events[w:]].mean()
        within_across_fixk[left_out, p] = within - across
        
        perm_lengths = np.random.permutation(event_lengths)
        events = np.zeros(nTR, dtype=np.int)
        events[np.cumsum(perm_lengths[:-1])] = 1
        events = np.cumsum(events)
    print('Subj ' + str(left_out+1) + ': within vs across = ' + str(within_across_fixk[left_out,0]))


# In[66]:


if ipynby==1: # plot within versus across boundary
    f, ax = plt.subplots(1,1, figsize = (8,8))
    rankd=stats.percentileofscore(within_across_fixk[:,1:].mean(0),within_across_fixk[:,0].mean(0))
    yl='''True vs %s scrambled event 
    boundaries, fixed k, no smoothing''' %iters
    vplot(within_across_fixk[:,0].mean(0),within_across_fixk[:,1:].mean(0),yl,rankd,1,0)
    f.tight_layout()
    fign=ss_dir+'pics/HMMFixk-WithinVAcross-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[195]:


#### best k 
# now set up outer and inner loop to find the best k via cross-validation
n_splits_inner = 4
subj_id_all = np.array([i for i in range(nS)])# set up the nested cross validation template
ks = np.arange(2,19)# set up outer loop loo structure
loglik=np.zeros((nS,nS,len(ks)))
best_ll=np.zeros((nS,nS))
best_k=np.zeros((nS,n_splits_inner))
best_k_subj=np.zeros((nS))
lls_inner=np.zeros((nS))
lls_outer=np.zeros((nS))
events_subj=np.zeros((nS,T))
best_k_ll_subj=np.zeros((nS))
all_ll=np.zeros((n_splits_inner,len(ks)))
loo_outer=LeaveOneOut()
loo_outer.get_n_splits(subj_id_all)
for subj_id_train_outer, subj_id_test_outer in loo_outer.split(subj_id_all):
    print("Outer:\tTrain:", subj_id_train_outer, "Test:", subj_id_test_outer)
    # set up inner loop loo structure
    subj_id_all_inner = subj_id_all[subj_id_train_outer]
    kf = KFold(n_splits=n_splits_inner)
    kf.get_n_splits(subj_id_train_outer)
    jj=0
    print('Inner:')
    for subj_id_train_inner, subj_id_test_inner in kf.split(subj_id_all_inner):
        # inplace update the ids w.r.t. to the inner training set
        subj_id_train_inner = subj_id_all_inner[subj_id_train_inner]
        subj_id_test_inner = subj_id_all_inner[subj_id_test_inner]
        print("-Train:", subj_id_train_inner, "Test:", subj_id_test_inner, ', now try different k...')
    
        D_train = D[:,:,subj_id_train_inner]
        D_val = D[:,:,subj_id_test_inner]
        D_test = D[:,:,subj_id_test_outer]

        for knum in range(0,len(ks)):
            ev = brainiak.eventseg.event.EventSegment(ks[knum])
            ev.fit(D_train.mean(2).T)
            segments,ll=ev.find_events(D_val.mean(2).T)
            print('log likelihood for k=%s is %s' %(ks[knum],ll))
            loglik[subj_id_test_outer,jj,knum] = ll
        all_ll[jj,:]=loglik[subj_id_test_outer,jj,:]
        best_ll[subj_id_test_outer,jj] = np.max(loglik[subj_id_test_outer,jj,:])
        jj=jj+1
    mean_all_ll=np.mean(all_ll,axis=0)
    fill3=mean_all_ll.argsort()
    print(fill3)
    fill3=fill3[len(ks)-1]
    print(fill3)
    best_k_subj[subj_id_test_outer]=ks[fill3]
    ev = brainiak.eventseg.event.EventSegment(ks[fill3])
    ev.fit(D_train.mean(2).T)
    segments,ll=ev.find_events(D_test.mean(2).T)     
    lls_outer[subj_id_test_outer]=np.max(ev.ll_)
    events_subj[subj_id_test_outer,:] = np.argmax(ev.segments_[0], axis=1)
print(lls_outer)
print(events_subj)


# In[196]:


if ipynby==1: # plot predicted event boundaries for every subject for ROI / game
    f, ax = plt.subplots(1,1, figsize = (10,8))
    this_img=events_subj
    ax.imshow(this_img,interpolation='none', cmap='viridis', aspect='auto')
    ax.set_ylabel('Subject')
    ax.set_title('Predicted event boundaries')
    ax.set_xlabel('TRs')
    ax.vlines(true_edges,0-0.5,this_img.shape[0]-0.5)
    f.tight_layout()
    fign=ss_dir+'pics/HMMBestk-EventBoundariesAllSubs-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[197]:


#plot best_k versus the actual number of possessions
best_k_xsub=np.max(events_subj,axis=1)+1
if ipynby==1:
    f, ax = plt.subplots(1,1, figsize = (5,5))
    histbins=ax.hist(best_k_xsub)
    print(np.max(histbins[0]))
    ax.set_ylabel('Best k across subjects')
    ax.vlines(maxedges,0,np.max(histbins[0]),linestyles='dashed')
    f.tight_layout()
    fign=ss_dir+'pics/HMMBestk-Hist-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[198]:


#find agreement between possession boundaries and the HMM, using best K  
bounds_subj=np.zeros((nS,T))
bounds_subj_smooth=np.zeros((nS,T))
for subj in range(0,nS):
    bs=np.where(np.diff(events_subj[subj,:]))
    bounds_subj[subj,bs] = 1
    bounds_subj_smooth[subj,:]=running_mean(bounds_subj[subj,:],smoothf)
sum_bounds=np.mean(bounds_subj,axis=0)
sum_bounds_smooth=running_mean(sum_bounds,smoothf)
#run correlation between possession boundaries and the HMM
fill=np.corrcoef(sum_bounds,true_edges_cont)
agree_r=fill[1,0]
fill=np.corrcoef(sum_bounds_smooth,true_edges_cont)
smooth_agree_r=fill[1,0]
if ipynby==1:
    f, ax = plt.subplots(2,1, figsize = (14,10)) 
    ax[0].plot(sum_bounds)
    ax[0].set_ylabel('Agreement across subjects')
    ax[0].set_title('Correlation = %s' %agree_r)
    ax[0].vlines(true_edges,0,np.max(sum_bounds),linestyles='dashed')
    ax[1].plot(sum_bounds_smooth)
    ax[1].set_ylabel('Smoothed agreement across subjects')
    ax[1].set_title('Correlation = %s' %smooth_agree_r)
    ax[1].set_xlabel('TRs')
    ax[1].vlines(true_edges,0,np.max(sum_bounds_smooth),linestyles='dashed')
    f.tight_layout()
    fign=ss_dir+'pics/HMMagreementCorrs-bestk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[199]:


#run correlation against amount of surprise rather than binary possession change(1/0)
ind=true_edges_cont_sur>0
fill=np.corrcoef(sum_bounds[ind],true_edges_cont_sur[ind])
agree_sur_r=fill[1,0]
if np.isnan(agree_sur_r):
    agree_sur_r=0
ind_s=true_edges_cont_sur_smooth>0
fill=np.corrcoef(sum_bounds_smooth[ind_s],true_edges_cont_sur_smooth[ind_s])
smooth_agree_sur_r=fill[1,0]
if np.isnan(smooth_agree_sur_r):
    smooth_agree_sur_r=0
if ipynby==1: 
    f, ax = plt.subplots(2,1, figsize = (14,10))  
    ax[0].plot(sum_bounds[ind])
    ax[0].set_ylabel('''Agreement across subjects, 
    surprise, fix k''')
    ax[0].set_title('Correlation = %s' %agree_sur_r)
    ax[0].plot((0.01+np.max(sum_bounds[ind]))*(true_edges_cont_sur[ind]/np.max(true_edges_cont_sur[ind])))
    ax[1].plot(sum_bounds_smooth[ind_s])
    ax[1].set_ylabel('''Agreement across subjects, 
    surprise, fixed k, smooth''')
    ax[1].set_title('Correlation = %s' %smooth_agree_sur_r)
    ax[1].set_xlabel('>0 surprise time points')
    ax[1].plot(np.max(sum_bounds_smooth[ind_s])*
               (true_edges_cont_sur_smooth[ind_s]/np.max(true_edges_cont_sur_smooth[ind_s])))
    f.tight_layout()
    fign=ss_dir+'pics/HMMagreementCorrs-Sur-bestk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[200]:


#test against null distribution of scrambled boundaries, using best K
agree_iter_r=np.zeros((iters))
smooth_agree_iter_r=np.zeros((iters))
agree_iter_sur_r=np.zeros((iters))
smooth_agree_iter_sur_r=np.zeros((iters))
jj=0
#for iter in range(0,iters):
while jj<1000:
    rps=np.random.permutation(poss_sizes.shape[0])
    true_edges_rp=np.cumsum(poss_sizes[rps]) #assign new edges
    true_edges_rp=np.hstack([0,true_edges_rp[:-1]]) #add 0 at beginning and cut last row to make similar to true_edges
    true_edges_rp_cont=[]
    true_edges_rp_cont=np.zeros((T))
    true_edges_rp_cont[true_edges_rp[1:].astype(int)]=1 #put in continuous space
    true_edges_rp_cont_smooth=running_max(true_edges_rp_cont,smoothf)
    true_edges_rp_cont_sur=np.zeros((T))
    true_edges_rp_cont_sur[true_edges_rp[1:].astype(int)]=true_edges_sur[1:] 
    true_edges_rp_cont_sur_smooth=running_max(true_edges_rp_cont_sur,smoothf)
    
    fill=np.corrcoef(sum_bounds,true_edges_rp_cont)
    agree_iter_r[jj]=fill[1,0]
    if np.isnan(agree_iter_r[jj]):
        agree_iter_r[jj]=0
    fill=np.corrcoef(sum_bounds_smooth,true_edges_rp_cont_smooth)
    smooth_agree_iter_r[jj]=fill[1,0]
    if np.isnan(smooth_agree_iter_r[jj]):
        smooth_agree_iter_r[jj]=0
    ind=true_edges_rp_cont_sur>0
    fill=np.corrcoef(sum_bounds[ind],true_edges_rp_cont_sur[ind])
    agree_iter_sur_r[jj]=fill[1,0]
    if np.isnan(agree_iter_sur_r[jj]):
        agree_iter_sur_r[jj]=0
    ind_s=true_edges_rp_cont_sur_smooth>0
    fill=np.corrcoef(sum_bounds_smooth[ind_s],true_edges_rp_cont_sur_smooth[ind_s])
    smooth_agree_iter_sur_r[jj]=fill[1,0]
    if np.isnan(smooth_agree_iter_sur_r[jj]):
        smooth_agree_iter_sur_r[jj]=0
    if ~np.isnan(agree_iter_sur_r[jj]):
        jj=jj+1


# In[201]:


agree_rankd=stats.percentileofscore(agree_iter_r,agree_r)
smooth_agree_rankd=stats.percentileofscore(smooth_agree_iter_r,smooth_agree_r)
agree_sur_rankd=stats.percentileofscore(agree_iter_sur_r,agree_sur_r)
smooth_agree_sur_rankd=stats.percentileofscore(smooth_agree_iter_sur_r,smooth_agree_sur_r)
if ipynby==1:
    f, ax = plt.subplots(2,1, figsize = (5,10))
    yl='''True vs %s scrambled event 
    boundaries, best k, no smoothing''' %iters
    vplot(agree_r,agree_iter_r,yl,agree_rankd,2,0)
    yl='''True vs %s scrambled event 
    boundaries, best k, smoothing''' %iters
    vplot(smooth_agree_r,smooth_agree_iter_r,yl,smooth_agree_rankd,2,1)
    f.tight_layout()
    fign=ss_dir+'pics/HMMBestk-AgreementVsIters-bestk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)
    f, ax = plt.subplots(2,1, figsize = (5,10))
    yl='''True vs %s scrambled event 
    boundaries, surprise, 
    best k, no smoothing''' %iters
    vplot(agree_sur_r,agree_iter_sur_r,yl,agree_sur_rankd,2,0)
    yl='''True vs %s scrambled event 
    boundaries, surprise, 
    best k, smoothing''' %iters
    vplot(smooth_agree_sur_r,smooth_agree_iter_sur_r,yl,smooth_agree_sur_rankd,2,1)
    f.tight_layout()
    fign=ss_dir+'pics/HMMBestk-SurAgreementVsIters-bestk-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)
    


# In[202]:


within_across = np.zeros((nS, nPerm+1))
for left_out in range(nS):
    k=int(best_k_xsub[left_out])
    ev = brainiak.eventseg.event.EventSegment(k)
    ev.fit(D[:,:,np.arange(nS) != left_out].mean(2).T)
    events = np.argmax(ev.segments_[0], axis=1)

    corrs = np.zeros(nTR-w) # Compute correlations separated by w in time
    for t in range(nTR-w):
        corrs[t] = pearsonr(D[:,t,left_out],D[:,t+w,left_out])[0]
    _, event_lengths = np.unique(events, return_counts=True)

    # Compute within vs across boundary correlations, for real and permuted bounds
    np.random.seed(0)
    for p in range(nPerm+1):
        within = corrs[events[:-w] == events[w:]].mean()
        across = corrs[events[:-w] != events[w:]].mean()
        within_across[left_out, p] = within - across
        
        perm_lengths = np.random.permutation(event_lengths)
        events = np.zeros(nTR, dtype=np.int)
        events[np.cumsum(perm_lengths[:-1])] = 1
        events = np.cumsum(events)
    print('Subj ' + str(left_out+1) + ': within vs across = ' + str(within_across[left_out,0]))
    


# In[203]:


if ipynby==1: # plot within versus across boundary
    f, ax = plt.subplots(1,1, figsize = (8,8))
    rankd=stats.percentileofscore(within_across[:,1:].mean(0),within_across[:,0].mean(0))
    yl='''True vs %s scrambled event 
    boundaries, best k, no smoothing''' %iters
    vplot(within_across[:,0].mean(0),within_across[:,1:].mean(0),yl,rankd,1,0)
    f.tight_layout()
    fign=ss_dir+'pics/HMMBestk-WithinVAcross-%s-%s.pdf' %(task_name,parcels[parcel_sel+1])
    f.savefig(fign)


# In[205]:


# save outputs as .mat
mat_fname=dir_out+'event_hmm_out-%s-%s-%s-%s.mat' %(n_trunc,hrshiftval,task_name,parcels[parcel_sel+1])
scipy.io.savemat(mat_fname,{'smoothf': smoothf,'maxedges': maxedges,'true_edges': true_edges,'true_edges_edge': true_edges_edge,'true_edges_sur': true_edges_sur,
    'bounds_subj_fixk': bounds_subj_fixk,'bounds_subj_smooth_fixk': bounds_subj_smooth_fixk,'sum_bounds_fixk': sum_bounds_fixk,'sum_bounds_smooth_fixk': sum_bounds_smooth_fixk,
    'true_edges_cont': true_edges_cont,'true_edges_cont_smooth': true_edges_cont_smooth,'agree_fixk_r': agree_fixk_r,
    'smooth_agree_fixk_r': smooth_agree_fixk_r,'true_edges_cont_sur': true_edges_cont_sur,'true_edges_cont_sur_smooth': true_edges_cont_sur_smooth,
    'agree_fixk_sur_r': agree_fixk_sur_r,'smooth_agree_fixk_sur_r': smooth_agree_fixk_sur_r,'agree_fixk_iter_r': agree_fixk_iter_r,
    'smooth_agree_fixk_iter_sur_r': smooth_agree_fixk_iter_sur_r,'within_across_fixk': within_across_fixk,'bounds_subj': bounds_subj,
    'bounds_subj_smooth': bounds_subj_smooth,
    'events_subj': events_subj,'lls_outer': lls_outer,'sum_bounds': sum_bounds,'sum_bounds_smooth': sum_bounds_smooth,'agree_r': agree_r,
    'smooth_agree_r': smooth_agree_r,'agree_sur_r': agree_sur_r,'smooth_agree_sur_r': smooth_agree_sur_r,
    'agree_iter_r': agree_iter_r,'smooth_agree_iter_r': smooth_agree_iter_r,
    'agree_iter_sur_r': agree_iter_sur_r,'smooth_agree_iter_sur_r': smooth_agree_iter_sur_r,'within_across': within_across,
    'agree_fixk_rankd': agree_fixk_rankd,'smooth_agree_fixk_rankd': smooth_agree_fixk_rankd,
    'agree_fixk_sur_rankd': agree_fixk_sur_rankd,'smooth_agree_fixk_sur_rankd': smooth_agree_fixk_sur_rankd,
    'agree_rankd': agree_rankd,'smooth_agree_rankd': smooth_agree_rankd,
    'agree_sur_rankd': agree_sur_rankd,'smooth_agree_sur_rankd': smooth_agree_sur_rankd
})


# In[ ]:




