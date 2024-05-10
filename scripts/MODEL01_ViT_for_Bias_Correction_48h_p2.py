import os
import sys
import time
import h5py
import numpy as np
from glob import glob

# ------------------------------------------------------- #
# Turn-off warnings
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ------------------------------------------------------- #
# Turn-off tensoflow-specific warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

# ------------------------------------------------------- #
# Import customized modules and settings
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import model_utils as mu
import verif_utils as vu

mu.set_seeds(888)

def to_precip(x):
    x[x<0] = 0
    return 10*(np.exp(x) - 1)

def verif_func(model, valid_GEFS, valid_CCPA_full, land_mask_CCPA):
    
    L_valid = valid_GEFS.shape[0]
    N_leads = EN = valid_GEFS.shape[1]
    EN = valid_GEFS.shape[2]

    shape_full = (L_valid, N_leads, EN, 224, 464)
    GEFS_full = np.empty(shape_full)
    pred_ = np.empty(valid_GEFS.shape[:-1]+(4,))

    for iens in range(EN):
        pred_[:, :, iens, ...] = model.predict(valid_GEFS[:, :, iens, ...], verbose=0)
        
    for ilead in range(N_leads):
        for iens in range(EN):
            GEFS_full[:, ilead, iens, ...] = decoder.predict(pred_[:, ilead, iens, ...], 
                                                             verbose=0)[..., 0]
    GEFS_full = to_precip(GEFS_full)
    CRPS = np.empty((L_valid, N_leads, 224, 464)); CRPS[...] = np.nan

    for ilead in range(N_leads):
        crps_ilead, _, _ = vu.CRPS_2d(valid_CCPA_full[:, ilead, ...], 
                                      GEFS_full[:, ilead, ...], land_mask=land_mask_CCPA)
        CRPS[:, ilead, ...] = crps_ilead
    return np.nanmean(CRPS)

# Hyperparameters
filter_nums = [64, 128] # number of convolution kernels per down-/upsampling layer 
latent_dim = 4 # number of latent feature channels
activation = 'gelu' # activation function
num_embeddings = 128 #128 # number of the VQ codes

input_size = (224, 464, 1) # size of MRMS input
latent_size = (14, 29, latent_dim) # size of compressed latent features

drop_encode = False
drop_decode = False

model_name_encoder_load = model_dir+'models/VQ_VAE_encoder_stack1_tune0'
model_name_decoder_load = model_dir+'models/VQ_VAE_decoder_stack1_tune0'

encoder = mu.VQ_VAE_encoder(input_size, filter_nums, latent_dim, num_embeddings, activation, drop_encode)

W_old = mu.dummy_loader(model_name_encoder_load)
encoder.set_weights(W_old)

decoder = mu.VQ_VAE_decoder(latent_size, filter_nums, activation, drop_decode)

W_old = mu.dummy_loader(model_name_decoder_load)
decoder.set_weights(W_old)

# ------------------------------------------------------- #
# Hyperparameters
# input size for the 48h models
latent_size = (14, 29, 4)
input_size = (8, 14, 29, 4)
output_size = (8, 14, 29, 4)

N_ens = 31

# # 0-48 hr settings
# lead_name = '0_48'
# ilead_start = 0
# ilead_end = 8
# N_leads = ilead_end - ilead_start
# pad_timelag = 2

# # 54-96 hr settings
# lead_name = '54_96'
# ilead_start = 8
# ilead_end = 16
# N_leads = ilead_end - ilead_start
# pad_timelag = 2

# 102-144 hr settings
lead_name = '102_144'
ilead_start = 16
ilead_end = 24
N_leads = ilead_end - ilead_start
pad_timelag = 2

# ============================= #
# Tuned hyperparameters
patch_size = (1, 1, 1) # (time, space, space)
N_heads = 4
N_layers = 8
project_dim = 128
# ============================= #

load_weights = True

# location of the previous weights
model_name_load = model_dir+'baseline/ViT3d_{}_depth{}_patch{}{}{}_dim{}_heads{}_tune2'.format(
    lead_name, N_layers, patch_size[0], patch_size[1], patch_size[2], project_dim, N_heads)
# location for saving new weights
model_name_save = model_dir+'baseline/ViT3d_{}_depth{}_patch{}{}{}_dim{}_heads{}_tune2'.format(
    lead_name, N_layers, patch_size[0], patch_size[1], patch_size[2], project_dim, N_heads)

# Training setups
epochs = 9999
batch_size = 64 #64
N_batch = 32
lrs = mu.cosine_schedule(N_batch, l_min=1e-6, l_max=5e-5)

aug_timelag = True
aug_revert = True

if aug_timelag:
    N_pad = pad_timelag # + pad_timelag0

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    land_mask_CCPA = h5io['land_mask_CCPA'][...]
    
land_mask_CCPA = land_mask_CCPA == 1.0
ccpa_shape = land_mask_CCPA.shape

# ------------------------------------------------------- #
# Validation set
BATCH_dir = camp_dir+'BATCH_ViT_members_opt/'
filenames = sorted(glob(BATCH_dir+'*npy'))

L_valid = 50
filenames_valid = filenames[::2][:L_valid] #[::10]

valid_GEFS = np.empty((L_valid, N_leads, N_ens)+input_size[1:])
valid_CCPA = np.empty((L_valid, N_leads,)+output_size[1:])

valid_GEFS_raw = np.empty((L_valid, N_leads, N_ens)+(224, 464))
valid_CCPA_true = np.empty((L_valid, N_leads)+(224, 464))

for i, name_ in enumerate(filenames_valid):
    temp_data = np.load(name_, allow_pickle=True)[()]
    valid_GEFS[i, ...] = temp_data['GEFS_embed'][ilead_start:ilead_end, ..., 0:4]
    valid_CCPA[i, ...] = temp_data['CCPA_embed'][ilead_start:ilead_end, ...]
    valid_GEFS_raw[i, ...] = temp_data['GEFS_raw'][ilead_start:ilead_end, ...]
    valid_CCPA_true[i, ...] = temp_data['CCPA_true'][ilead_start:ilead_end, ...]

valid_CCPA_true = to_precip(valid_CCPA_true)

valid_GEFS = valid_GEFS[:, :, ::7, ...]
valid_GEFS_raw = valid_GEFS_raw[:, :, ::7, ...]

BATCH_dir = camp_dir+'BATCH_ViT/'
filename_train = sorted(glob(BATCH_dir+'*npy'))
filename_train = list(set(filename_train) - set(filenames_valid))
L_train = len(filename_train)

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

# ------------------------------------------------------- #
# Training loop
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    
    model = mu.ViT3d_corrector(input_size, output_size, patch_size, project_dim, N_layers, N_heads)
    
    model.compile(loss=keras.losses.mean_absolute_error, 
                  optimizer=keras.optimizers.Adam(learning_rate=5e-5))
    
    # load weights
    if load_weights:
        W_old = mu.dummy_loader(model_name_load)
        model.set_weights(W_old)
        
    # ----------------------------------------------- #
    # Major training loop + training batch generation
    
    batch_GEFS = np.empty((batch_size, N_leads+N_pad,)+latent_size)
    batch_GEFS[...] = np.nan
    batch_CCPA = np.empty((batch_size, N_leads+N_pad,)+latent_size)
    batch_CCPA[...] = np.nan

    batch_GEFS_aug = np.empty((batch_size, N_leads,)+latent_size)
    batch_GEFS_aug[...] = np.nan
    batch_CCPA_aug = np.empty((batch_size, N_leads,)+latent_size)
    batch_CCPA_aug[...] = np.nan
    
    for i in range(epochs):
        
        print('epoch = {}'.format(i))
        
        if i == 0:
            record = verif_func(model, valid_GEFS, valid_CCPA_true, land_mask_CCPA)
            print('Initial validation loss: {}'.format(record))
        
        start_time = time.time()
        for j in range(N_batch):

            tf.keras.backend.set_value(model.optimizer.learning_rate, lrs[j])
            
            inds_rnd = du.shuffle_ind(L_train)
            inds_ = inds_rnd[:batch_size]

            ## Note: always train with first few days 
            for k, ind in enumerate(inds_):
                # import batch data
                name_ = filename_train[ind]
                temp_data = np.load(name_, allow_pickle=True)[()]
                batch_GEFS[k, ...] = temp_data['GEFS_embed'][0:8+pad_timelag, ...]
                batch_CCPA[k, ...] = temp_data['CCPA_embed'][0:8+pad_timelag, ...]

            if aug_timelag:
                for k in range(batch_size):
                    i_start = np.random.randint(0, N_pad)
                    batch_GEFS_aug[k, ...] = batch_GEFS[k, i_start:i_start+N_leads, ...]
                    batch_CCPA_aug[k, ...] = batch_CCPA[k, i_start:i_start+N_leads, ...]

            if aug_revert:
                for k in range(batch_size):
                    i_revert = np.random.randint(0, 4)
                    if i_revert == 4:
                        batch_GEFS_aug[k, ...] = batch_GEFS_aug[k, ::-1, ...]
                        batch_CCPA_aug[k, ...] = batch_CCPA_aug[k, ::-1, ...]

            if (aug_timelag is False) and (aug_revert is False):
                batch_GEFS_aug = batch_GEFS[:, pad_timelag0:-pad_timelag, ...]
                batch_CCPA_aug = batch_CCPA[:, pad_timelag0:-pad_timelag, ...]
                
            if np.sum(np.isnan(batch_GEFS_aug)) > 0:
                raise
                
            model.train_on_batch(batch_GEFS_aug, batch_CCPA_aug)
            
        # on epoch-end
        record_temp = verif_func(model, valid_GEFS, valid_CCPA_true, land_mask_CCPA)
    
        if record - record_temp > min_del:
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            print("Save to {}".format(model_name_save))
            model.save(model_name_save)
            
        else:
            print('Validation loss {} NOT improved'.format(record_temp))
        
        print("--- %s seconds ---" % (time.time() - start_time))
        # mannual callbacks









