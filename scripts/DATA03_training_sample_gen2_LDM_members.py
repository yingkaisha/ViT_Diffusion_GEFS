'''
Generate training batches for LDM
'''

import os
import sys
import time
import h5py
import numpy as np
from glob import glob
from datetime import datetime, timedelta

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

def norm_precip(x):
    return np.log(0.1*x+1)
    
# ------------------------------------------------------- #
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

base = datetime(year, 1, 1)

if year % 4 == 0:
    N_days = 366
else:
    N_days = 365

date_list = [base + timedelta(days=d) for d in range(N_days)]

# Hyperparameters
filter_nums = [64, 128] # number of convolution kernels per down-/upsampling layer 
latent_dim = 4 # number of latent feature channels
activation = 'gelu' # activation function
num_embeddings = 128 #128 # number of the VQ codes

input_size = (224, 464, 1) # size of MRMS input
latent_size = (14, 29, latent_dim) # size of compressed latent features

drop_encode = True
drop_decode = True

# location for saving new weights
model_name_encoder_load = model_dir+'models/VQ_VAE_encoder_stack1_tune0'
model_name_decoder_load = model_dir+'models/VQ_VAE_decoder_stack1_tune0'

model_encoder = mu.VQ_VAE_encoder(input_size, filter_nums, latent_dim, num_embeddings, activation, drop_encode)

W_old = mu.dummy_loader(model_name_encoder_load)
model_encoder.set_weights(W_old)


def ViT_pred(x, model_48, model_96, model_144):
    Y_pred_48 = model_48.predict(x[0:8, ...][None, ...], verbose=0)
    Y_pred_96 = model_96.predict(x[8:16, ...][None, ...], verbose=0)
    Y_pred_144 = model_144.predict(x[16:24, ...][None, ...], verbose=0)
    Y_pred = np.concatenate((Y_pred_48, Y_pred_96, Y_pred_144), axis=1)
    return Y_pred[0, ...]

# The tensor size of embedded CCPA and GEFS ensemble mean 
latent_size = (14, 29, 4)
# input size for the 48h models
input_size = (8,) + latent_size

# patch size
patch_size = (1, 1, 1) # (time, space, space)

N_heads = 4
N_layers = 8
project_dim = 128

model_name_load_48h = model_dir+'baseline/ViT3d_0_48_depth8_patch111_dim128_heads4_tune2'
model_name_load_96h = model_dir+'baseline/ViT3d_54_96_depth8_patch111_dim128_heads4_tune2'
model_name_load_144h = model_dir+'baseline/ViT3d_102_144_depth8_patch111_dim128_heads4_tune2'

model_48h = mu.ViT3d_corrector(input_size, input_size, patch_size, project_dim, N_layers, N_heads)
W_old = mu.dummy_loader(model_name_load_48h)
model_48h.set_weights(W_old)

model_96h = mu.ViT3d_corrector(input_size, input_size, patch_size, project_dim, N_layers, N_heads)
W_old = mu.dummy_loader(model_name_load_96h)
model_96h.set_weights(W_old)

model_144h = mu.ViT3d_corrector(input_size, input_size, patch_size, project_dim, N_layers, N_heads)
W_old = mu.dummy_loader(model_name_load_144h)
model_144h.set_weights(W_old)

BATCH_dir = camp_dir+'BATCH_LDM_member/'
batch_name = 'LDM_{}_{}.npy' #.format(datetime)

LEADs = np.arange(6, 144+6, 6) # forecast lead times
N_leads = len(LEADs)

grid_shape = (N_leads, 14, 29, 4)
ccpa_shape = (224, 464)

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    land_mask_CCPA = h5io['land_mask_CCPA'][...]
land_mask_CCPA = land_mask_CCPA == 0

filename_ccpa = camp_dir+'CCPA/CCPA_y{}.hdf'

with h5py.File(filename_ccpa.format(year), 'r') as h5io:
    CCPA_base = h5io['CCPA'][...]
L_base = len(CCPA_base)

# forecast lead times can exceed one year
N_beyond = 10
N_total = L_base + N_beyond
with h5py.File(filename_ccpa.format(year+1), 'r') as h5io:
    CCPA_extra = h5io['CCPA'][:N_beyond, ...]

CCPA = np.concatenate((CCPA_base, CCPA_extra), axis=0)
CCPA = norm_precip(CCPA) # <------ normed CCPA
CCPA[:, :, land_mask_CCPA] = 0.0

CCPA_true = np.empty((24, 224, 464))
GEFS_embed = np.empty((24, 14, 29, 4))
CCPA_embed = np.empty((24, 14, 29, 4))
ViT_embed = np.empty((24, 14, 29, 4))

filename_gefs = camp_dir+'GFS_reforecast/GEFS_{}_y{}.hdf'
ens_names = ['mean', 'c00', 'p01', 'p02', 'p03', 'p04',]

filename_gefs_AVE = camp_dir+'GFS_reforecast/GEFS_AVE_y{}.hdf'

with h5py.File(filename_gefs_AVE.format(year), 'r') as h5io:
    GEFS_AVE = h5io['GEFS_APCP'][:, :N_leads, ...] 

GEFS_AVE = norm_precip(GEFS_AVE)
GEFS_AVE[:, :, land_mask_CCPA] = 0.0

for ens in ens_names:
    if ens == 'mean':
        GEFS = GEFS_AVE
    else:
        with h5py.File(filename_gefs.format(ens, year), 'r') as h5io:
            try:
                GEFS = h5io['GEFS_APCP'][...]
            except:
                GEFS = h5io['APCP'][...]
        
    GEFS = norm_precip(GEFS)
    GEFS[:, :, land_mask_CCPA] = 0.0

    for d, dt in enumerate(date_list):
        dt_str = datetime.strftime(dt, '%Y%m%d')
        apcp = GEFS[d, ...]
        
        GEFS_embed[...] = np.nan
        CCPA_embed[...] = np.nan
        CCPA_embed[...] = np.nan
        ViT_embed[...] = np.nan

        for ilead, lead in enumerate(LEADs):
            d_ = lead // 24
            day = d + d_
            ind_hour = lead % 24
            ind_hour = int(ind_hour/6)

            CCPA_true[ilead, ...] = CCPA[day, ind_hour, ...]
            ccpa_input = CCPA[day, ind_hour, ...][None, ..., None]
            CCPA_embed[ilead, ...] = model_encoder.predict(ccpa_input, verbose=0)[0, ...]

        gefs_input = apcp[..., None]
        GEFS_embed = model_encoder.predict(gefs_input, verbose=0)
            
        ViT_embed = ViT_pred(GEFS_embed, model_48h, model_96h, model_144h)
        
        if np.sum(np.isnan(CCPA_true)) + np.sum(np.isnan(CCPA_embed)) + \
           np.sum(np.isnan(GEFS_embed)) + np.sum(np.isnan(ViT_embed)) == 0:
            data_save = {}
            data_save['CCPA_true'] = CCPA_true #normed CCPA
            data_save['CCPA_embed'] = CCPA_embed
            data_save['GEFS_embed'] = GEFS_embed
            data_save['ViT_embed'] = ViT_embed
            save_name_ = BATCH_dir+batch_name.format(dt_str, ens)
            print(save_name_)
            np.save(save_name_, data_save)
        else:
            print('Found NaN')






