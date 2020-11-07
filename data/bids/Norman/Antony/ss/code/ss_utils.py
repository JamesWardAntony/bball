import numpy as np 
import scipy.io
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from copy import deepcopy

# data path to ss dataset 
ss_dir = '/jukebox/norman/jantony/surprisesuspense/'
ss_bids_dir = '/jukebox/norman/jantony/surprisesuspense/data/bids/Norman/Antony/ss/'

# constants for the ss dataset (localizer)
ss_all_ROIs = ['ses01brain', 'V1', 'bilateral_NAcc', 'bilateral_HC', 'bilateral_frontal_inf-orbital']

ss_TR = 1
ss_hrf_lag = 5  # In seconds what is the lag between a stimulus onset and the peak bold response

run_names = ['view','recall']
n_runs = [3]
ss_tngs = 9

# TRs_run = [311, 406, 214, 166, 406, 406, 214]

def get_MNI152_template(dim_x, dim_y, dim_z):
    """get MNI152 template used in fmrisim
    Parameters
    ----------
    dim_x: int
    dim_y: int
    dim_z: int
        - dims set the size of the volume we want to create
    
    Return
    -------
    MNI_152_template: 3d array (dim_x, dim_y, dim_z)
    """
    # Import the fmrisim from BrainIAK
    import brainiak.utils.fmrisim as sim 
    # Make a grey matter mask into a 3d volume of a given size
    dimensions = np.asarray([dim_x, dim_y, dim_z])
    _, MNI_152_template = sim.mask_brain(dimensions)
    return MNI_152_template

def load_ss_mask(ROI_name, sub):
    """Load the mask for the ss data 
    Parameters
    ----------
    ROI_name: string
    sub: string 
    
    Return
    ----------
    the requested mask
    """    
    #assert ROI_name in ss_all_ROIs
    maskdir = (ss_bids_dir + "derivatives/firstlevel/" + sub + "/masks/")
    # load the mask
    maskfile = (maskdir + sub + "_%s.nii.gz" % (ROI_name))
    mask = nib.load(maskfile)
    print("Loaded %s mask" % (ROI_name))
    return mask


def load_ss_epi_data(sub, run):
    # Load MRI file (in Nifti format) of one localizer run
    epi_in = (ss_bids_dir +  
              "derivatives/fmriprep/%s/ses-01/func/%s_ses-01_task-view_run-0%i_space-T1w_desc-preproc_bold.nii.gz" % (sub,sub,run))
    epi_data = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data


def mask_data(epi_data, mask): 
    """mask the input data with the input mask 
    Parameters
    ----------
    epi_data
    mask
    
    Return
    ----------
    masked data
    """    
    nifti_masker = NiftiMasker(mask_img=mask)
    epi_masked_data = nifti_masker.fit_transform(epi_data);
    return epi_masked_data


def scale_data(data): 
    data_scaled = preprocessing.StandardScaler().fit_transform(data)
    return data_scaled

""""""

# Make a function to load the mask data
def load_data(directory, subject_name, mask_name='', num_runs=2, zscore_data=False):
    
    # Cycle through the masks
    print ("Processing Start ...")
    
    # If there is a mask supplied then load it now
    if mask_name is '':
        mask = None
    else:
        mask = load_ss_mask(mask_name, subject_name)

    # Cycle through the runs
    for run in range(1, num_runs + 1):
        epi_data = load_ss_epi_data(subject_name, run)
        
        # Mask the data if necessary
        if mask_name is not '':
            epi_mask_data = mask_data(epi_data, mask).T
        else:
            # Do a whole brain mask 
            if run == 1:
                # Compute mask from epi
                mask = compute_epi_mask(epi_data).get_fdata()  
            else:
                # Get the intersection mask 
                # (set voxels that are within the mask on all runs to 1, set all other voxels to 0)   
                mask *= compute_epi_mask(epi_data).get_fdata()  
            
            # Reshape all of the data from 4D (X*Y*Z*time) to 2D (voxel*time): not great for memory
            epi_mask_data = epi_data.get_fdata().reshape(
                mask.shape[0] * mask.shape[1] * mask.shape[2], 
                epi_data.shape[3]
            )

        # Transpose and z-score (standardize) the data  
        if zscore_data == True:
            scaler = preprocessing.StandardScaler().fit(epi_mask_data.T)
            preprocessed_data = scaler.transform(epi_mask_data.T)
        else:
            preprocessed_data = epi_mask_data.T
        
        # Concatenate the data
        if run == 1:
            concatenated_data = preprocessed_data
        else:
            concatenated_data = np.hstack((concatenated_data, preprocessed_data))
    
    # Apply the whole-brain masking: First, reshape the mask from 3D (X*Y*Z) to 1D (voxel). 
    # Second, get indices of non-zero voxels, i.e. voxels inside the mask. 
    # Third, zero out all of the voxels outside of the mask.
    if mask_name is '':
        mask_vector = np.nonzero(mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], ))[0]
        concatenated_data = concatenated_data[mask_vector, :]
        
    # Return the list of mask data
    return concatenated_data, mask

# Create a function to shift the size
def shift_timing(label_TR, TR_shift_size):
    
    # Create a short vector of extra zeros
    zero_shift = np.zeros((TR_shift_size, 1))

    # Zero pad the column from the top.
    label_TR_shifted = np.vstack((zero_shift, label_TR))

    # Don't include the last rows that have been shifted out of the time line.
    label_TR_shifted = label_TR_shifted[0:label_TR.shape[0],0]
    
    return label_TR_shifted

# Extract bold data for non-zero labels.
def reshape_data(label_TR_shifted, masked_data_all):
    label_index = np.nonzero(label_TR_shifted)
    label_index = np.squeeze(label_index)
    
    # Pull out the indexes
    indexed_data = np.transpose(masked_data_all[:,label_index])
    nonzero_labels = label_TR_shifted[label_index] 
    
    return indexed_data, nonzero_labels

def normalize(bold_data_, run_ids):
    """normalized the data within each run
    
    Parameters
    --------------
    bold_data_: np.array, n_stimuli x n_voxels
    run_ids: np.array or a list
    
    Return
    --------------
    normalized_data
    """
    scaler = StandardScaler()
    data = []
    for r in range(ss_n_runs):
        data.append(scaler.fit_transform(bold_data_[run_ids == r, :]))
    normalized_data = np.vstack(data)
    return normalized_data
    
    
def decode(X, y, cv_ids, model): 
    """
    Parameters
    --------------
    X: np.array, n_stimuli x n_voxels
    y: np.array, n_stimuli, 
    cv_ids: np.array - n_stimuli, 
    
    Return
    --------------
    models, scores
    """
    scores = []
    models = []
    ps = PredefinedSplit(cv_ids)
    for train_index, test_index in ps.split():
        # split the data 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit the model on the training set 
        model.fit(X_train, y_train)
        # calculate the accuracy for the hold out run
        score = model.score(X_test, y_test)
        # save stuff 
        models.append(deepcopy(model))
        scores.append(score)
    return models, scores
