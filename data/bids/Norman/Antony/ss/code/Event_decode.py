#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this script is for decoding various characteristics about the games, including: 
# Game being watched, Team out of 18 with the ball, Possession out of 157, 
#Whether The possession goes left Or right, whether the winning team is in possession of the ball, 
# whether there is a free-throw or real gameplay
ipynby=0 #python notebook or not
if ipynby==0:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cond', type=str)
    args = parser.parse_args()
    print(args.cond)
    cond=args.cond
if ipynby==1:
    cond=0
subs=['01','02','03','04','05','06','07','08','09']#
n_trunc=3 # Number of volumes to trim/truncate
all_task_names = ['view_sb0']
print(all_task_names)
ROIs = ['V1','A1','ang','precun','HC','DMN','PMC','PCC','vmPFC','NAcc']
ROI_sel = int(cond)
print(ROIs[ROI_sel])


# In[2]:


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
import scipy.stats
from brainiak import image, io
#from isc_standalone import isc
from brainiak.isc import isc, isfc, permutation_isc, timeshift_isc, phaseshift_isc
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.pyplot as plt 
from numpy.polynomial.polynomial import polyfit
get_ipython().run_line_magic('autosave', '5')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = 'white', context='talk', font_scale=1, rc={"lines.linewidth": 2})


# In[3]:


from ss_utils import ss_dir, ss_bids_dir, ss_TR, ss_hrf_lag, run_names, n_runs
from ss_utils import load_ss_epi_data, load_ss_mask, mask_data, load_data
#mask_name=ss_bids_dir+'derivatives/freesurfer/fsaverage/mri/brainmask.mgz'
results_path=ss_bids_dir+'derivatives/secondlevel/'
firstlevel_dir=ss_bids_dir+'derivatives/firstlevel/'
analysis_dir=ss_dir+'analysis/'
mat_fname=analysis_dir+'d_event_mat.mat'
mat_contents = scipy.io.loadmat(mat_fname)
event_mat = mat_contents['event_mat'] 
event_fl = mat_contents['event_fl'] #1st TR for event matrix
newposs = mat_contents['newposs']
oldposs = mat_contents['oldposs']
oldposs2 = np.zeros((oldposs.shape[0]))
for i in range(0,oldposs.shape[0]):
    oldposs2[i] = oldposs[i]
print(oldposs.shape)
print(oldposs2.shape)
n_subjs_total = len(subs)
ses='ses-01'

dir_out = results_path + 'decode/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
    print('Dir %s created ' % dir_out)


# In[4]:


def get_file_names(data_dir_, task_name_,n_trunc_, verbose = False):
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
    # Collect all file names 
    for subj in range(1, n_subjs_total+1): 
        fname = os.path.join(
            data_dir_, 'sub-0%s/sub-0%s_task-%s_trim%s.nii.gz' % (subj, subj, task_name_,n_trunc_))
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


# In[5]:


#find avg brain mask across subjects!
mask_imgs=[]
for subj in range(1,n_subjs_total+1):
    mask_fold = ss_bids_dir + 'derivatives/firstlevel/sub-0%s/masks/' % subj
    mask_name = mask_fold + 'sub-0%s_%s_brain.nii.gz' % (subj, ses)
    mask_imgs.append(mask_name)
    brain_mask1 = io.load_boolean_mask(mask_name)


# In[6]:


# intersect 3 view brain masks    
#save x-sub avg mask
avg_mask_name = results_path + 'avg_brain_mask.nii.gz'

#load in mask as boolean
brain_mask = io.load_boolean_mask(avg_mask_name)# Load the brain mask
coords = np.where(brain_mask)# Get the list of nonzero voxel coordinates
brain_nii = nib.load(avg_mask_name)# Load the brain nii image
print(brain_nii.shape)


# In[ ]:


# load the functional data 
fnames = {}
images = {}
masked_images = {}
bold = {}
n_subjs = {}

for task_name in all_task_names: 
    fnames[task_name] = get_file_names(firstlevel_dir, task_name,n_trunc)
    images[task_name] = io.load_images(fnames[task_name]) 
    #images[task_name] = images[task_name].T #?? 
    masked_images[task_name] = image.mask_images(images[task_name], brain_mask) 
    # Concatenate all of the masked images across participants  
    bold[task_name] = image.MaskedMultiSubjectData.from_masked_images(
        masked_images[task_name], len(fnames[task_name]))
    # print(np.shape(bold[task_name]))
    # Convert nans into zeros
    bold[task_name][np.isnan(bold[task_name])] = 0
    # compute the group assignment label 
    n_subjs_this_task = np.shape(bold[task_name])[-1]
    n_subjs[task_name] = np.shape(bold[task_name])[-1]
    # print('Data loaded: {} \t shape: {}' .format(task_name, np.shape(bold[task_name])))


# In[ ]:


# Compute spatial pattern at every time point within select ROIs
# Get a list of ROIs. 
sub = 'sub-01' # sample subject for ROI
# Collect all ROIs 
all_roi_nii = {}
all_roi_masker = {}
# Cycle through the masks
for mask_counter in range(0,len(ROIs)):
    out_dir= ss_bids_dir + 'derivatives/firstlevel/%s/' % sub
    roi_fn = out_dir + '%s_task-view_event_%s.mat' % (sub,ROIs[mask_counter])
    roi_name = ROIs[mask_counter]
    if os.path.exists(roi_fn):
        roi_fn = out_dir + '%s_task-view_event_%s.mat' % (sub,ROIs[mask_counter])
        mat_contents = scipy.io.loadmat(roi_fn)
        epi_masked_data = mat_contents['epi_masked_data'] #design matrix for GLM
        print('File found, loading %s' % (roi_fn))
        all_roi_masker[roi_name] = epi_masked_data
    else:
        roi_fn = out_dir + 'masks/%s_%s.nii.gz' % (sub,ROIs[mask_counter])
        #roi_fn = out_dir + '%s_task-view_event_%s.mat' % (sub,ROIs[mask_counter])
        # Load roi nii file 
        roi_nii = nib.load(roi_fn)
        all_roi_nii[roi_name] = roi_nii
        # Make roi maskers
        all_roi_masker[roi_name] = NiftiMasker(mask_img=roi_nii)
# print(all_roi_masker)


# In[ ]:


# Make a function to load data for one ROI
def load_roi_data(roi_name): 
    roi_masker = all_roi_masker[roi_name] # Pick a roi masker
    bold_roi = {task_name:[] for i, task_name in enumerate(all_task_names)} # Preallocate 
    for task_name in all_task_names: # Gather data
        for subj_id in range(n_subjs[task_name]):
            # Get the data for task t, subject s 
            nii_t_s = nib.load(fnames[task_name][subj_id])
            bold_roi[task_name].append(roi_masker.fit_transform(nii_t_s))
        # Reformat the data to std form 
        bold_roi[task_name] = np.transpose(np.array(bold_roi[task_name]), [1,2,0])
    return bold_roi


# In[ ]:


# Load ROI 
for j in range(ROI_sel,ROI_sel+1): #enumerate(ROIs): , roi_name
    roi_name=ROIs[j]
    bold_roi = load_roi_data(roi_name) # Load data


# In[ ]:


sub_game_vox={}
bincol=0
tngs=9
sub_game_vox[task_name]=np.zeros((tngs,len(subs),bold_roi[task_name].shape[1]))
for game in range(0,tngs):
    ind=event_mat[:,bincol]==game+1
    #print(ind)
    bold_temp=[]
    bold_temp=bold_roi[task_name][ind,:,:]
    #print(bold_temp.shape)
    bold_temp_mean=np.mean(bold_temp,axis=0)
    #print(bold_temp_mean.shape)
    sub_game_vox[task_name][game,:,:]=bold_temp_mean.T
game_corrs=np.zeros((len(subs),tngs,tngs))
game_ranks=np.zeros((len(subs),tngs))
game_ranks_top=np.zeros((len(subs),tngs))
for subject in range(0,len(subs)):
    other_subs = np.arange(0,len(subs)-1)
    other_subs[subject:]=other_subs[subject:]+1
    sub_pattern = sub_game_vox[task_name][:,subject,:]
    mean_pattern=np.mean(sub_game_vox[task_name][:,other_subs,:],axis=1)
    for game in range(0,tngs): #game the mean pattern is drawn from
        for game2 in range(0,tngs): #game the subject is on
            fill=np.corrcoef(mean_pattern[game,:],sub_pattern[game2,:])
            game_corrs[subject,game,game2]=fill[1,0]
        fill2=game_corrs[subject,game,:].argsort()[::-1]
        game_ranks[subject,game]=np.argwhere(fill2==game)+1
        if game_ranks[subject,game]==1: #if top-ranked game
            game_ranks_top[subject,game]=1
mean_game_corrs=np.mean(game_corrs,axis=0)

f, ax = plt.subplots(1,1, figsize=(10,8))
x = np.arange(1,tngs+1) # x-coordinates of your bars
this_img = mean_game_corrs 
tn="game by game correlation for %s" %ROIs[j]
plt.imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
plt.title(tn)
plt.xticks(x-1,x)
plt.yticks(x-1,x)
ax.set_ylabel("left out subject-game number") #left out subject
ax.set_xlabel("mean of other subjects-game number")
cbar = plt.colorbar()
cbar.ax.set_ylabel("R")
f.tight_layout()
fign=ss_dir+'pics/game-by-game-corr-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

w = 0.8    # bar width
x2=np.arange(1,tngs+2)-0.5
y = game_ranks
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(tngs+1)/2+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('rank of game (of all games)')
ax.set_xlabel('game number')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofgame-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(game_ranks,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((game_ranks.shape[0]))*(tngs+1)/2, y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(tngs+1)/2+0*x2,'--k')
plt.title('game, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanrank-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,tngs+1)
x2=np.arange(1,tngs+2)-0.5
y = game_ranks_top
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,1/tngs+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('classifier performance')
ax.set_xlabel('game number')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofgame-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)    

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(game_ranks_top,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((game_ranks_top.shape[0]))*(1/tngs), y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(1/tngs)+0*x2,'--k')
plt.title('game, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanrank-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


#decode which team has the ball
team_poss=(event_mat[:,0]-1)*2+event_mat[:,4]+1
#plt.plot(team_poss)

sub_team_vox={}
sub_team_vox[task_name]=np.zeros((tngs*2,len(subs),bold_roi[task_name].shape[1]))
for team in range(0,tngs*2):
    ind=team_poss==team+1
    bold_temp=[]
    bold_temp=bold_roi[task_name][ind,:,:]
    bold_temp_mean=np.mean(bold_temp,axis=0)
    sub_team_vox[task_name][team,:,:]=bold_temp_mean.T
team_corrs=np.zeros((len(subs),tngs*2,tngs*2))
team_ranks=np.zeros((len(subs),tngs*2))
team_ranks_top=np.zeros((len(subs),tngs*2))
for subject in range(0,len(subs)):
    other_subs = np.arange(0,len(subs)-1)
    other_subs[subject:]=other_subs[subject:]+1
    sub_pattern = sub_team_vox[task_name][:,subject,:]
    mean_pattern=np.mean(sub_team_vox[task_name][:,other_subs,:],axis=1)
    for team in range(0,tngs*2): #team the mean pattern is drawn from
        for team2 in range(0,tngs*2): #team the subject is on
            fill=np.corrcoef(mean_pattern[team,:],sub_pattern[team2,:])
            team_corrs[subject,team,team2]=fill[1,0]
        fill2=team_corrs[subject,team,:].argsort()[::-1]
        team_ranks[subject,team]=np.argwhere(fill2==team)+1
        if team_ranks[subject,team]==1:
            team_ranks_top[subject,team]=1
mean_team_corrs=np.mean(team_corrs,axis=0)

f, ax = plt.subplots(1,1, figsize=(10,8))
x = np.arange(1,tngs*2+1) # x-coordinates of your bars
this_img = mean_team_corrs 
tn="team by team possession correlation for %s" %ROIs[j]
plt.imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
plt.title(tn)
plt.xticks(x-1,x)
plt.yticks(x-1,x)
ax.set_ylabel("left out subject-team possession number") #left out subject
ax.set_xlabel("mean of other subjects-team possession number")
cbar = plt.colorbar()
cbar.ax.set_ylabel("R")
f.tight_layout()
fign=ss_dir+'pics/team-by-team-corr-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x2=np.arange(1,tngs*2+2)-0.5
y = team_ranks
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(tngs*2+1)/2+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('rank of team (of all teams)')
ax.set_xlabel('team number')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofteam-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(team_ranks,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((team_ranks.shape[0]))*(tngs*2+1)/2, y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(tngs*2+1)/2+0*x2,'--k')
plt.title('team possession, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanteamrank-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,tngs*2+1) # x-coordinates of your bars
x2=np.arange(1,tngs*2+2)-0.5
y = team_ranks_top
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(1/(tngs*2))+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('classifier performance')
ax.set_xlabel('team number')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofteam-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)    

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(team_ranks_top,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((team_ranks_top.shape[0]))*(1/(tngs*2)), y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,1/(tngs*2)+0*x2,'--k')
plt.title('team possession, classifier performance, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('classifier performance')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanteamrank-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


plt.plot(team_poss)
plt.plot(oldposs2)
ind=[]
team=0
ind=team_poss==team+1
plt.plot(ind)

max_both_team_poss=int(np.nanmax(oldposs2[ind]))
k=0
for poss in range(0,max_both_team_poss):
    ind=[]
    ind=(team_poss==team+1) & (oldposs2==poss+1)
    if np.sum(ind)>0:
        k=k+1
max_team_poss=k
print(oldposs2)
print(team_poss)


# In[ ]:


#mean_team_poss_corrs=np.zeros((tngs*2))
f, ax = plt.subplots(3,6, figsize=(16,8))
mean_team_poss_ranks = np.zeros((tngs*2,len(subs)))
team_posses = np.zeros((tngs*2))
for team in range(0,tngs*2):
    ind=[]
    ind=team_poss==team+1
    max_both_team_poss=int(np.nanmax(oldposs2[ind]))
    k=0
    for poss in range(0,max_both_team_poss):
        ind=[]
        ind=(team_poss==team+1) & (oldposs2==poss+1)
        if np.sum(ind)>0:
            k=k+1
    max_team_poss=k
    sub_team_poss_vox={}
    sub_team_poss_vox[task_name]=np.zeros((max_team_poss,len(subs),bold_roi[task_name].shape[1]))
    k=0
    for poss in range(0,max_both_team_poss):
        ind=[]
        ind=(team_poss==team+1) & (oldposs2==poss+1)
        if np.sum(ind)>0:
            bold_temp=[]
            bold_temp=bold_roi[task_name][ind,:,:]
            bold_temp_mean=np.mean(bold_temp,axis=0)
            sub_team_poss_vox[task_name][k,:,:]=bold_temp_mean.T
            k=k+1
    team_poss_corrs=np.zeros((len(subs),max_team_poss,max_team_poss))
    team_poss_ranks=np.zeros((len(subs),max_team_poss))
    team_poss_ranks_top=np.zeros((len(subs),max_team_poss))
    for subject in range(0,len(subs)):
        other_subs = np.arange(0,len(subs)-1)
        other_subs[subject:]=other_subs[subject:]+1
        sub_pattern = sub_team_poss_vox[task_name][:,subject,:]
        mean_pattern=np.mean(sub_team_poss_vox[task_name][:,other_subs,:],axis=1)
        for poss in range(0,max_team_poss): #team the mean pattern is drawn from
            for poss2 in range(0,max_team_poss): #team the subject is on
                fill=np.corrcoef(mean_pattern[poss,:],sub_pattern[poss2,:])
                team_poss_corrs[subject,poss,poss2]=fill[1,0]
            fill2=team_poss_corrs[subject,poss,:].argsort()[::-1]
            team_poss_ranks[subject,poss]=np.argwhere(fill2==poss)+1
            if team_poss_ranks[subject,poss]==1:
                team_poss_ranks_top[subject,poss]=1
    mean_team_poss_corrs=np.mean(team_poss_corrs,axis=0)
    mean_team_poss_ranks[team,:]=np.mean(team_poss_ranks,axis=1)
    team_posses[team]=max_team_poss

#for team in range(0,tngs*2):    
    row = int(np.floor(team/6))
    col = team%6
    x = np.arange(1,max_team_poss+1) # x-coordinates of your bars
    this_img = mean_team_poss_corrs 
    tn="possession by possession correlation for each team, %s" %ROIs[j]
    plt.sca(ax[row,col])
    #cmap = colors.ListedColormap(['b','g','y','r'])
    #bounds=[-0.3,-0.1,0.1,0.3]
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(this_img,cmap="viridis",vmin=-.3,vmax=.3,origin="upper",interpolation="none",aspect="auto")
    #plt.pcolor(X, Y, v, cmap=cm)
    #ax[row,col].imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
    ax[row,col].xaxis.set_visible(False)
    ax[row,col].yaxis.set_visible(False)
    #plt.xticks(x-1,x)
    #plt.yticks(x-1,x)
    if team==6:
        ax[row,col].set_ylabel("left out subject-possession number for each team") #left out subject
    if team==3:
        ax[row,col].set_title(tn)
    if team==14:    
        ax[row,col].set_xlabel("mean of other subjects-possession number for each team")
    cbar = plt.colorbar(ax=ax[row,col])
    #cbar = fig.colorbar(im, cax=cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0.5,1.5,2.5,3.5],)
    cbar.ax.set_ylabel("R")
    
f.tight_layout()
fign=ss_dir+'pics/poss-by-poss-corr-each-team-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(mean_team_poss_ranks,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((mean_team_poss_ranks.shape[0]))*(np.mean(team_posses)+1)/2, y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(np.mean(team_posses)+1)/2+0*x2,'--k')
plt.title('possession within team, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanteampossrank-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


#decode possession within game
poss_num=np.zeros((event_mat.shape[0]))
for tr in range(0,event_mat.shape[0]):
    if np.isnan(newposs[tr])==0:
        poss_num[tr]=newposs[tr]
print(poss_num)


# In[ ]:


#plt.plot(poss_num)
max_poss_num=int(np.nanmax(poss_num))
print(max_poss_num)
sub_poss_vox={}
sub_poss_vox[task_name]=np.zeros((max_poss_num,len(subs),bold_roi[task_name].shape[1]))
sus_poss=np.zeros((max_poss_num))
for poss in range(0,max_poss_num):
    ind=poss_num==poss+1
    bold_temp=[]
    bold_temp=bold_roi[task_name][ind,:,:]
    bold_temp_mean=np.mean(bold_temp,axis=0)
    sub_poss_vox[task_name][poss,:,:]=bold_temp_mean.T 
    sus_poss[poss]=np.nanmean(event_mat[ind,10])
poss_corrs=np.zeros((len(subs),max_poss_num,max_poss_num))
poss_ranks=np.zeros((len(subs),max_poss_num))
poss_ranks_top=np.zeros((len(subs),max_poss_num))
for subject in range(0,len(subs)):
    other_subs = np.arange(0,len(subs)-1)
    other_subs[subject:]=other_subs[subject:]+1
    sub_pattern = sub_poss_vox[task_name][:,subject,:]
    mean_pattern=np.mean(sub_poss_vox[task_name][:,other_subs,:],axis=1)
    for poss in range(0,max_poss_num): #team the mean pattern is drawn from
        for poss2 in range(0,max_poss_num): #team the subject is on
            fill=np.corrcoef(mean_pattern[poss,:],sub_pattern[poss2,:])
            poss_corrs[subject,poss,poss2]=fill[1,0]
        fill2=poss_corrs[subject,poss,:].argsort()[::-1]
        poss_ranks[subject,poss]=np.argwhere(fill2==poss)+1
        if poss_ranks[subject,poss]==1:
            poss_ranks_top[subject,poss]=1
mean_poss_corrs=np.mean(poss_corrs,axis=0)


# In[ ]:


f, ax = plt.subplots(1,1, figsize=(10,8))
x = np.arange(1,max_poss_num+1,step=10) # x-coordinates of your bars
this_img = mean_poss_corrs 
tn="possession by possession correlation for %s" %ROIs[j]
plt.imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
plt.title(tn)
plt.xticks(x-1,x)
plt.yticks(x-1,x)
ax.set_ylabel("left out subject-possession number") #left out subject
ax.set_xlabel("mean of other subjects-possession number")
cbar = plt.colorbar()
cbar.ax.set_ylabel("R")
f.tight_layout()
fign=ss_dir+'pics/poss-by-poss-corr-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

f, ax = plt.subplots(1,1, figsize=(10,8))
this_img = mean_poss_corrs[sus_poss.argsort(),:]

tn="possession by possession correlation for %s" %ROIs[j]
plt.imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
plt.title(tn)
plt.xticks(x-1,x)
plt.yticks(x-1,x)
ax.set_ylabel("left out subject-possession number (sorted by suspense)") #left out subject number
ax.set_xlabel("mean of other subjects-possession number (sorted by suspense)")
cbar = plt.colorbar()
cbar.ax.set_ylabel("R")
f.tight_layout()
fign=ss_dir+'pics/poss-by-poss-corr-sus-sort-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,max_poss_num+1) # x-coordinates of your bars
x2=np.arange(1,max_poss_num+2)-0.5
y = poss_ranks
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(max_poss_num+1)/2+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('rank of possession (of all possessions)')
ax.set_xlabel('possession number')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofposs-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(poss_ranks,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((poss_ranks.shape[0]))*(max_poss_num+1)/2, y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(max_poss_num+1)/2+0*x2,'--k')
plt.title('possession, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanpossrank-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,max_poss_num+1) # x-coordinates of your bars
x2=np.arange(1,max_poss_num+2)-0.5
y = poss_ranks_top
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,1/max_poss_num+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('classifier performance')
ax.set_xlabel('possession number')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofposs-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(poss_ranks_top,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((poss_ranks_top.shape[0]))*(1/max_poss_num), y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(1/max_poss_num)+0*x2,'--k')
plt.title('possession, classifier performance, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('classifier performance')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanpossrank-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


x=sus_poss
y=np.mean(poss_ranks,axis=0)
sus_poss_rank_r,sus_poss_rank_p=scipy.stats.pearsonr(x,y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.scatter(x,y)
ax.set_xlabel('suspense (for possession)')
ax.set_ylabel('mean rank')
plt.title('sus vs. poss decode (r=%s,p=%s)' %(sus_poss_rank_r,sus_poss_rank_p))
f.tight_layout()
fign=ss_dir+'pics/suspossrankcorr-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


#decode whether it's a team going left or right on the screen
lr=event_mat[:,7]
lrconds=2
#plt.plot(lr)
sub_lr_vox={}
sub_lr_vox[task_name]=np.zeros((lrconds,len(subs),bold_roi[task_name].shape[1]))
sub_lr_vox2={}
sub_lr_vox2[task_name]=np.zeros((lrconds,len(subs),bold_roi[task_name].shape[1]))
lr_corrs=np.zeros((len(subs),tngs,lrconds,lrconds))
lr_ranks=np.zeros((len(subs),tngs,lrconds))
lr_ranks_top=np.zeros((len(subs),tngs,lrconds))
for game in range(0,tngs):
    
    for direction in range(0,lrconds):
        ind=(lr==direction) & (event_mat[:,0]==game+1)
        ind2=(lr==direction) & (event_mat[:,0]!=game+1)
        #print(ind)
        #print(ind2)
        bold_temp=[]
        bold_temp=bold_roi[task_name][ind,:,:]
        bold_temp_mean=np.mean(bold_temp,axis=0)
        sub_lr_vox[task_name][direction,:,:]=bold_temp_mean.T 
        bold_temp=[]
        bold_temp=bold_roi[task_name][ind2,:,:]
        bold_temp_mean=np.mean(bold_temp,axis=0)
        sub_lr_vox2[task_name][direction,:,:]=bold_temp_mean.T
    for subject in range(0,len(subs)):
        other_subs = np.arange(0,len(subs)-1)
        other_subs[subject:]=other_subs[subject:]+1
        sub_pattern = sub_lr_vox[task_name][:,subject,:]
        mean_pattern=np.mean(sub_lr_vox2[task_name][:,other_subs,:],axis=1)
        for direction in range(0,lrconds): #team the mean pattern is drawn from
            for direction2 in range(0,lrconds): #team the subject is on
                fill=np.corrcoef(mean_pattern[direction,:],sub_pattern[direction2,:])
                lr_corrs[subject,game,direction,direction2]=fill[1,0]
            fill2=lr_corrs[subject,game,direction,:].argsort()[::-1]
            lr_ranks[subject,game,direction]=np.argwhere(fill2==direction)+1
            if lr_ranks[subject,game,direction]==1:
                lr_ranks_top[subject,game,direction]=1
mean_lr_corrs=np.mean(lr_corrs,axis=0)
print(mean_lr_corrs.shape)
mean_lr_corrs=np.mean(mean_lr_corrs,axis=0)
print(mean_lr_corrs.shape)
lr_ranks=np.mean(lr_ranks,axis=1)
lr_ranks_top=np.mean(lr_ranks_top,axis=1)


# In[ ]:


f, ax = plt.subplots(1,1, figsize=(10,8))
x = np.arange(1,lrconds+1) # x-coordinates of your bars
this_img = mean_lr_corrs 
tn="direction by direction correlation for %s" %ROIs[j]
plt.imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
plt.title(tn)
plt.xticks(x-1,x)
plt.yticks(x-1,x)
ax.set_ylabel("left out subject-direction") #left out subject
ax.set_xlabel("mean of other subjects-direction")
cbar = plt.colorbar()
cbar.ax.set_ylabel("R")
f.tight_layout()
fign=ss_dir+'pics/dir-by-dir-corr-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x2=np.arange(1,lrconds+2)-0.5
y = lr_ranks
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(lrconds+1)/2+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('rank of direction (of both directions)')
ax.set_xlabel('direction number')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofdir-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(lr_ranks,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((lr_ranks.shape[0]))*(lrconds+1)/2, y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(lrconds+1)/2+0*x2,'--k')
plt.title('direction, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meandirrank-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,lrconds+1) # x-coordinates of your bars
x2=np.arange(1,lrconds+2)-0.5
y = lr_ranks_top
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,1/lrconds+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('classifier performance')
ax.set_xlabel('direction')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofdir-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(lr_ranks_top,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((lr_ranks_top.shape[0]))*(1/lrconds), y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(1/lrconds)+0*x2,'--k')
plt.title('direction, classifier performance, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('classifier performance')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meandirrank-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


#decode whether the winning team has the ball
wp=event_mat[:,6]
wpconds=2
#plt.plot(wp)
sub_wp_vox={}
sub_wp_vox[task_name]=np.zeros((wpconds,len(subs),bold_roi[task_name].shape[1]))
sub_wp_vox2={}
sub_wp_vox2[task_name]=np.zeros((wpconds,len(subs),bold_roi[task_name].shape[1]))
wp_corrs=np.zeros((len(subs),tngs,wpconds,wpconds))
wp_ranks=np.zeros((len(subs),tngs,wpconds))
wp_ranks_top=np.zeros((len(subs),tngs,wpconds))
for game in range(0,tngs): 
    for winposs in range(0,wpconds):
        ind=(wp==winposs) & (event_mat[:,0]==game+1)
        ind2=(wp==winposs) & (event_mat[:,0]!=game+1)
        #print(ind)
        #print(ind2)
        bold_temp=[]
        bold_temp=bold_roi[task_name][ind,:,:]
        bold_temp_mean=np.mean(bold_temp,axis=0)
        sub_wp_vox[task_name][winposs,:,:]=bold_temp_mean.T 
        bold_temp=[]
        bold_temp=bold_roi[task_name][ind2,:,:]
        bold_temp_mean=np.mean(bold_temp,axis=0)
        sub_wp_vox2[task_name][winposs,:,:]=bold_temp_mean.T
    for subject in range(0,len(subs)):
        other_subs = np.arange(0,len(subs)-1)
        other_subs[subject:]=other_subs[subject:]+1
        sub_pattern = sub_wp_vox[task_name][:,subject,:]
        mean_pattern=np.mean(sub_wp_vox2[task_name][:,other_subs,:],axis=1)
        for winposs in range(0,wpconds): #team the mean pattern is drawn from
            for winposs2 in range(0,wpconds): #team the subject is on
                fill=np.corrcoef(mean_pattern[winposs,:],sub_pattern[winposs2,:])
                wp_corrs[subject,game,winposs,winposs2]=fill[1,0]
            fill2=wp_corrs[subject,game,winposs,:].argsort()[::-1]
            wp_ranks[subject,game,winposs]=np.argwhere(fill2==winposs)+1
            if wp_ranks[subject,game,winposs]==1:
                wp_ranks_top[subject,game,winposs]=1
mean_wp_corrs=np.mean(wp_corrs,axis=0)
print(mean_wp_corrs.shape)
mean_wp_corrs=np.mean(mean_wp_corrs,axis=0)
print(mean_wp_corrs.shape)
wp_ranks=np.mean(wp_ranks,axis=1)
wp_ranks_top=np.mean(wp_ranks_top,axis=1)


# In[ ]:


f, ax = plt.subplots(1,1, figsize=(10,8))
x = np.arange(1,wpconds+1) # x-coordinates of your bars
this_img = mean_wp_corrs 
tn="winposs by winposs correlation for %s" %ROIs[j]
plt.imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
plt.title(tn)
plt.xticks(x-1,x)
plt.yticks(x-1,x)
ax.set_ylabel("left out subject-winposs") #left out subject
ax.set_xlabel("mean of other subjects-winposs")
cbar = plt.colorbar()
cbar.ax.set_ylabel("R")
f.tight_layout()
fign=ss_dir+'pics/winposs-by-winposs-corr-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x2=np.arange(1,wpconds+2)-0.5
y = wp_ranks
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(wpconds+1)/2+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('rank of winposs (of both teams)')
ax.set_xlabel('winning or losing')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofwinposs-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(wp_ranks,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((wp_ranks.shape[0]))*(wpconds+1)/2, y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(wpconds+1)/2+0*x2,'--k')
plt.title('winposs, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanwinpossrank-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,wpconds+1) # x-coordinates of your bars
x2=np.arange(1,wpconds+2)-0.5
y = wp_ranks_top
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,1/wpconds+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('classifier performance')
ax.set_xlabel('possession (winning or losing)')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofwinposs-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(wp_ranks_top,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((wp_ranks_top.shape[0]))*(1/wpconds), y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(1/wpconds)+0*x2,'--k')
plt.title('winposs, classifier performance, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('classifier performance')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanwinpossrank-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


#decode whether it's a live game or a free throw
ft=event_mat[:,8]
ftconds=2
plt.plot(ft)
sub_ft_vox={}
sub_ft_vox[task_name]=np.zeros((ftconds,len(subs),bold_roi[task_name].shape[1]))
sub_ft_vox2={}
sub_ft_vox2[task_name]=np.zeros((ftconds,len(subs),bold_roi[task_name].shape[1]))
ft_corrs=np.zeros((len(subs),tngs,ftconds,ftconds))
ft_ranks=np.zeros((len(subs),tngs,ftconds))
ft_ranks_top=np.zeros((len(subs),tngs,ftconds))
for game in range(0,tngs): 
    for ftno in range(0,ftconds):
        ind=(ft==ftno) & (event_mat[:,0]==game+1)
        ind2=(ft==ftno) & (event_mat[:,0]!=game+1)
        #print(ind)
        #print(ind2)
        bold_temp=[]
        bold_temp=bold_roi[task_name][ind,:,:]
        bold_temp_mean=np.mean(bold_temp,axis=0)
        sub_ft_vox[task_name][ftno,:,:]=bold_temp_mean.T 
        bold_temp=[]
        bold_temp=bold_roi[task_name][ind2,:,:]
        bold_temp_mean=np.mean(bold_temp,axis=0)
        sub_ft_vox2[task_name][ftno,:,:]=bold_temp_mean.T
    for subject in range(0,len(subs)):
        other_subs = np.arange(0,len(subs)-1)
        other_subs[subject:]=other_subs[subject:]+1
        sub_pattern = sub_ft_vox[task_name][:,subject,:]
        mean_pattern=np.mean(sub_ft_vox2[task_name][:,other_subs,:],axis=1)
        for ftno in range(0,ftconds): #team the mean pattern is drawn from
            for ftno2 in range(0,ftconds): #team the subject is on
                fill=np.corrcoef(mean_pattern[ftno,:],sub_pattern[ftno2,:])
                ft_corrs[subject,game,ftno,ftno2]=fill[1,0]
            fill2=ft_corrs[subject,game,ftno,:].argsort()[::-1]
            ft_ranks[subject,game,ftno]=np.argwhere(fill2==ftno)+1
            if ft_ranks[subject,game,ftno]==1:
                ft_ranks_top[subject,game,ftno]=1
mean_ft_corrs=np.nanmean(ft_corrs,axis=0)
print(mean_ft_corrs.shape)
mean_ft_corrs=np.nanmean(mean_ft_corrs,axis=0)
print(mean_ft_corrs.shape)
ft_ranks=np.nanmean(ft_ranks,axis=1)
ft_ranks_top=np.nanmean(ft_ranks_top,axis=1)


# In[ ]:


f, ax = plt.subplots(1,1, figsize=(10,8))
x = np.arange(1,ftconds+1) # x-coordinates of your bars
this_img = mean_ft_corrs 
tn="free throw or no correlation for %s" %ROIs[j]
plt.imshow(this_img,cmap="viridis",origin="upper",interpolation="none",aspect="auto")
plt.title(tn)
plt.xticks(x-1,x)
plt.yticks(x-1,x)
ax.set_ylabel("left out subject-ftno") #left out subject
ax.set_xlabel("mean of other subjects-ftno")
cbar = plt.colorbar()
cbar.ax.set_ylabel("R")
f.tight_layout()
fign=ss_dir+'pics/ftno-by-ftno-corr-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x2=np.arange(1,ftconds+2)-0.5
y = ft_ranks
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(ftconds+1)/2+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('rank of free throw or no')
ax.set_xlabel('free throw or no')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofftno-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(ft_ranks,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((ft_ranks.shape[0]))*(ftconds+1)/2, y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(ftconds+1)/2+0*x2,'--k')
plt.title('free throw or no, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('mean rank')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanftnorank-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,ftconds+1) # x-coordinates of your bars
x2=np.arange(1,ftconds+2)-0.5
y = ft_ranks_top
f, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,1/ftconds+0*x2,'--k')
plt.xticks(x,x)
for ii in range(0,len(x)):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj,ii])
ax.set_ylabel('classifier performance')
ax.set_xlabel('game status (free throw or no)')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/rankofftno-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)

x = np.arange(1,1+1) # x-coordinates of your bars
x2=np.arange(1,1+2)-0.5
y = np.mean(ft_ranks_top,axis=1)
grt,grp=scipy.stats.ttest_rel(np.ones((ft_ranks_top.shape[0]))*(1/ftconds), y)
f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
ax.bar(x,np.mean(y,axis=0),yerr=np.std(y,axis=0),capsize=12,width=w,align='center', alpha=0.5)
ax.plot(x2,(1/ftconds)+0*x2,'--k')
plt.title('free throw or no, classifier performance, p value = %s' %grp)
for ii in range(0,1):
    for jj in range(0,len(subs)):
        ax.scatter(x[ii], y[jj])
ax.set_ylabel('classifier performance')
plt.show()
f.tight_layout()
fign=ss_dir+'pics/meanftnorank-top-%s-%s.png' %(task_name,ROIs[j])
f.savefig(fign)


# In[ ]:


mat_fname=dir_out+'event_decode_out-%s-%s-%s.mat' %(n_trunc,task_name,ROIs[j])
scipy.io.savemat(mat_fname,{'game_corrs': game_corrs,'game_ranks': game_ranks,'game_ranks_top':game_ranks_top,
                           'team_corrs': team_corrs,'team_ranks': team_ranks,'team_ranks_top':team_ranks_top,
                           'poss_corrs': poss_corrs,'poss_ranks': poss_ranks,'poss_ranks_top':poss_ranks_top,
                           'lr_corrs': lr_corrs,'lr_ranks': lr_ranks,'lr_ranks_top':lr_ranks_top,
                           'wp_corrs': wp_corrs,'wp_ranks': wp_ranks,'wp_ranks_top':wp_ranks_top,
                           'sub_game_vox': sub_game_vox,'sub_team_vox': sub_team_vox,'sub_poss_vox':sub_poss_vox,
                           'sub_lr_vox': sub_lr_vox,'sub_wp_vox': sub_wp_vox,'sub_ft_vox':sub_ft_vox,
                           'ft_corrs': ft_corrs,'ft_ranks': ft_ranks,'ft_ranks_top':ft_ranks_top})


# In[ ]:




