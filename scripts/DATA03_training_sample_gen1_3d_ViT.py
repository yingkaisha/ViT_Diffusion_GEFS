'''
Generate training samples for 3d ViT
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

# ------------------------------------------------------- #
# VQ-VAE encoder
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

# ------------------------------------------------------- #
# Sample gen information

BATCH_dir = camp_dir+'BATCH_ViT/'
batch_name = 'ViT_{}.npy' #.format(datetime)

LEADs = np.arange(6, 168+6, 6) # forecast lead times
N_leads = len(LEADs)

grid_shape = (N_leads, 14, 29, 4)
ccpa_shape = (224, 464)

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    land_mask_CCPA = h5io['land_mask_CCPA'][...]
land_mask_CCPA = np.logical_not(land_mask_CCPA)

# ------------------------------------------------------- #
# Datetime information

base = datetime(year, 1, 1)

if year % 4 == 0:
    N_days = 366
else:
    N_days = 365

date_list = [base + timedelta(days=d) for d in range(N_days)]

# ------------------------------------------------------- #
# CCPA and GEFS reforecast import

filename_gefs = camp_dir+'GFS_reforecast/GEFS_AVE_y{}.hdf'
filename_ccpa = camp_dir+'CCPA/CCPA_y{}.hdf'

with h5py.File(filename_gefs.format(year), 'r') as h5io:
    GEFS = h5io['GEFS_APCP'][...] 

with h5py.File(filename_ccpa.format(year), 'r') as h5io:
    CCPA_base = h5io['CCPA'][...]
L_base = len(CCPA_base)

# forecast lead times can exceed one year
N_beyond = 10
N_total = L_base + N_beyond
with h5py.File(filename_ccpa.format(year+1), 'r') as h5io:
    CCPA_extra = h5io['CCPA'][:N_beyond, ...]

CCPA = np.concatenate((CCPA_base, CCPA_extra), axis=0)

GEFS = norm_precip(GEFS)
CCPA = norm_precip(CCPA)

GEFS[:, :, land_mask_CCPA] = 0.0
CCPA[:, :, land_mask_CCPA] = 0.0

# ------------------------------------------------------- #
# The main sample generation loop

GEFS_embed = np.empty(grid_shape)
CCPA_embed = np.empty(grid_shape)

for d, dt in enumerate(date_list):
    dt_str = datetime.strftime(dt, '%Y%m%d')
    GEFS_embed[...] = np.nan
    CCPA_embed[...] = np.nan
    
    for ilead, lead in enumerate(LEADs):
        d_ = lead // 24; day = d + d_
        ind_hour = lead % 24; ind_hour = int(ind_hour/6)
        
        ccpa_input = CCPA[day, ind_hour, ...][None, ..., None]
        gefs_input = GEFS[d, ilead, ...][None, ..., None]

        if np.sum(np.isnan(ccpa_input)) + np.sum(np.isnan(gefs_input)) == 0:
            CCPA_embed[ilead, ...] = model_encoder.predict(ccpa_input, verbose=0)[0, ...]
            GEFS_embed[ilead, ...] = model_encoder.predict(gefs_input, verbose=0)[0, ...]

        else:
            CCPA_embed[ilead, ...] = np.nan
            GEFS_embed[ilead, ...] = np.nan

    if np.sum(np.isnan(CCPA_embed)) + np.sum(np.isnan(GEFS_embed)) == 0:        
        data_save = {}
        data_save['CCPA_embed'] = CCPA_embed
        data_save['GEFS_embed'] = GEFS_embed
        save_name_ = BATCH_dir+batch_name.format(dt_str)
        print(save_name_)
        np.save(save_name_, data_save)

