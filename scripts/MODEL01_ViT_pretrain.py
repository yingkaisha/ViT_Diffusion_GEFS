import os
import sys
import time
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

# ------------------------------------------------------- #
# Hyperparameters
# input size for the 48h models
latent_size = (14, 29, 4)
input_size = (8,) + latent_size

# 0-48 hr settings
lead_name = '0_48'
ilead_start = 0
ilead_end = 8
N_leads = ilead_end - ilead_start

# # 54-96 hr settings
# lead_name = '54_96'
# ilead_start = 8
# ilead_end = 16
# N_leads = ilead_end - ilead_start

# # 102-144 hr settings
# lead_name = '102_144'
# ilead_start = 16
# ilead_end = 24
# N_leads = ilead_end - ilead_start

# ============================= #
# Tuned hyperparameters
patch_size = (1, 1, 1) # (time, space, space)
N_heads = 4
N_layers = 8
project_dim = 128
# ============================= #

load_weights = True

# location of the previous weights
model_name_load = model_dir+'models/ViT3d_{}_depth{}_patch{}{}{}_dim{}_heads{}_tune'.format(
    lead_name, N_layers, patch_size[0], patch_size[1], patch_size[2], project_dim, N_heads)
# location for saving new weights
model_name_save = model_dir+'models/ViT3d_{}_depth{}_patch{}{}{}_dim{}_heads{}_tune'.format(
    lead_name, N_layers, patch_size[0], patch_size[1], patch_size[2], project_dim, N_heads)

# Training setups
lr = 1e-4
batch_size = 4 #64
N_batch = 32
epochs = 9999

aug_timelag = True
aug_revert = True

if aug_timelag:
    pad_timelag = 2
else:
    pad_timelag = 0

# ------------------------------------------------------- #
# Validation set
# BATCH_dir = camp_dir+'BATCH_ViT_OPT/'
BATCH_dir = camp_dir+'BATCH_ViT/'
filenames = sorted(glob(BATCH_dir+'*npy'))

L_valid = 500
filenames_valid = filenames[::10][:L_valid]

valid_GEFS = np.empty((L_valid, N_leads,)+latent_size)
valid_CCPA = np.empty((L_valid, N_leads,)+latent_size)

for i, name_ in enumerate(filenames_valid):
    temp_data = np.load(name_, allow_pickle=True)[()]
    valid_GEFS[i, ...] = temp_data['GEFS_embed'][ilead_start:ilead_end, ...]
    valid_CCPA[i, ...] = temp_data['CCPA_embed'][ilead_start:ilead_end, ...]

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
    
    model = mu.ViT3d_corrector(input_size, patch_size, project_dim, N_layers, N_heads)
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    # load weights
    if load_weights:
        W_old = mu.dummy_loader(model_name_load)
        model.set_weights(W_old)
        
    # ----------------------------------------------- #
    # Major training loop + training batch generation
    
    batch_GEFS = np.empty((batch_size, N_leads+pad_timelag,)+latent_size)
    batch_GEFS[...] = np.nan
    batch_CCPA = np.empty((batch_size, N_leads+pad_timelag,)+latent_size)
    batch_CCPA[...] = np.nan

    batch_GEFS_aug = np.empty((batch_size, N_leads,)+latent_size)
    batch_GEFS_aug[...] = np.nan
    batch_CCPA_aug = np.empty((batch_size, N_leads,)+latent_size)
    batch_CCPA_aug[...] = np.nan
    
    for i in range(epochs):
        
        print('epoch = {}'.format(i))
        if i == 0:
            Y_pred = model.predict(valid_GEFS)
            record = du.mean_absolute_error(valid_CCPA, Y_pred)
            print('Initial validation loss: {}'.format(record))
        
        start_time = time.time()
        for j in range(N_batch):
            
            inds_rnd = du.shuffle_ind(L_train)
            inds_ = inds_rnd[:batch_size]
            
            for k, ind in enumerate(inds_):
                # import batch data
                name_ = filename_train[ind]
                temp_data = np.load(name_, allow_pickle=True)[()]
                batch_GEFS[k, ...] = temp_data['GEFS_embed'][:N_leads+pad_timelag, ...]
                batch_CCPA[k, ...] = temp_data['CCPA_embed'][:N_leads+pad_timelag, ...]
                
            if aug_timelag:
                for k in range(batch_size):
                    i_start = np.random.randint(0, pad_timelag)
                    batch_GEFS_aug[k, ...] = batch_GEFS[k, i_start:i_start+N_leads, ...]
                    batch_CCPA_aug[k, ...] = batch_CCPA[k, i_start:i_start+N_leads, ...]
                    
            if aug_revert:
                for k in range(batch_size):
                    i_revert = np.random.randint(0, 4)
                    if i_revert == 4:
                        batch_GEFS_aug[k, ...] = batch_GEFS_aug[k, ::-1, ...]
                        batch_CCPA_aug[k, ...] = batch_CCPA_aug[k, ::-1, ...]

            if (aug_timelag is False) and (aug_revert is False):
                batch_GEFS_aug = batch_GEFS[:, :-pad_timelag, ...]
                batch_CCPA_aug = batch_CCPA[:, :-pad_timelag, ...]
                
            model.train_on_batch(batch_GEFS_aug, batch_CCPA_aug)
            
        # on epoch-end
        Y_pred = model.predict(valid_GEFS)
        record_temp = du.mean_absolute_error(valid_CCPA, Y_pred)
    
        if record - record_temp > min_del:
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            print("Save to {}".format(model_name_save))
            #model.save(model_name_save)
            
        else:
            print('Validation loss {} NOT improved'.format(record_temp))
        
        print("--- %s seconds ---" % (time.time() - start_time))
        # mannual callbacks


