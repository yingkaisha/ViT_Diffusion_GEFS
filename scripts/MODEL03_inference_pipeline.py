
import os
import sys
import time
import h5py
import numba as nb
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

#tf.config.run_functions_eagerly(True)
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('iday0', help='iday0')
parser.add_argument('iday1', help='iday1')
args = vars(parser.parse_args())

iday0 = int(args['iday0'])
iday1 = int(args['iday1'])

mu.set_seeds(888)

N_gen = 2 # generate # samples per GEFS ensemble member

# ------------------------------------------------------- #
# VQ-VAE
filter_nums = [64, 128] # number of convolution kernels per down-/upsampling layer 
latent_dim = 4 # number of latent feature channels
activation = 'gelu' # activation function
num_embeddings = 128 #128 # number of the VQ codes

input_size = (224, 464, 1) # size of MRMS input
latent_size = (14, 29, latent_dim) # size of compressed latent features

drop_encode = True
drop_decode = True

model_name_encoder_load = model_dir+'models/VQ_VAE_encoder_stack1_tune0'
model_name_decoder_load = model_dir+'models/VQ_VAE_decoder_stack1_tune0'

encoder = mu.VQ_VAE_encoder(input_size, filter_nums, latent_dim, num_embeddings, activation, drop_encode)
W_old = mu.dummy_loader(model_name_encoder_load)
encoder.set_weights(W_old)

decoder = mu.VQ_VAE_decoder(latent_size, filter_nums, activation, drop_decode)
W_old = mu.dummy_loader(model_name_decoder_load)
decoder.set_weights(W_old)

# ------------------------------------------------------- #
# Improved decoder blocks with elev and clim inputs
input_size = (224, 464, 3)
filter_nums = [32, 64, 32]

model_name_load = model_dir+'models/VAE_refine_tune/'

decoder_refine = mu.VQ_VAE_refine_blocks(input_size, filter_nums)
W_old = mu.dummy_loader(model_name_load)
decoder_refine.set_weights(W_old)


# ------------------------------------------------------- #
# ViT corrector
input_size = (8, 14, 29, 4)
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

# ------------------------------------------------------- #
# Diffusion model
input_size = (24, 14, 29, 4)

# model design
widths = [32, 64, 96, 128]
embedding_dims = 32
block_depth = 2

diffusion_steps = 100
min_signal_rate = 0.02
max_signal_rate = 0.95
ema = 0.999

# location of the previous weights
model_name_load = model_dir+'LDM_step{:02d}_base/'.format(diffusion_steps)

LDM = mu.DiffusionModel(input_size, input_size, input_size, 
                        diffusion_steps, min_signal_rate, max_signal_rate, 
                        embedding_dims, widths, block_depth, ema)

LDM.load_weights(model_name_load)


# ------------------------------------------------------- #
# Parameters & Helper functions

# the rescale value of LDM input
ccpa_min = -2.0606
ccpa_max = 1.6031

# latent space feature size
latent_size = (14, 29, 4)

# output size
ccpa_shape = (224, 464)

# Lead time information
LEADs = np.arange(6, 144+6, 6) # forecast lead times
N_leads = len(LEADs)

# helper functions

def norm_precip(x):
    return np.log(0.1*x+1)

def norm_elev(x):
    return x / np.nanmax(x)

def rescale(x, min_val=ccpa_min, max_val=ccpa_max):
    return ((x - min_val) / (max_val - min_val))* 2 - 1

def scaleback(x, min_val=ccpa_min, max_val=ccpa_max):
    return 0.5*(x+1)*(max_val-min_val) + min_val

def to_precip(x):
    x[x<0] = 0
    return 10*(np.exp(x) - 1)

def ViT_pred(x, model_48, model_96, model_144):
    Y_pred_48 = model_48.predict(x[:, 0:8, ...], verbose=0)
    Y_pred_96 = model_96.predict(x[:, 8:16, ...], verbose=0)
    Y_pred_144 = model_144.predict(x[:, 16:24, ...], verbose=0)
    Y_pred = np.concatenate((Y_pred_48, Y_pred_96, Y_pred_144), axis=1)
    return Y_pred

# ------------------------------------------------------- #
# Geographical & climatological info
# land mask and the actual shape of the CCPA CONUS domain
with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    land_mask_CCPA = h5io['land_mask_CCPA'][...]
    elev_CCPA = h5io['elev_CCPA'][...]
    
land_mask = land_mask_CCPA == 1.0
ocean_mask = land_mask_CCPA == 0.0

elev_CCPA[ocean_mask] = 0
elev_CCPA[elev_CCPA<0] = 0
elev_CCPA = norm_elev(elev_CCPA) # <-- normalization

with h5py.File(camp_dir+'CCPA/CCPA_CDFs_2002_2019.hdf', 'r') as h5io:
    CCPA_CDFs = h5io['CCPA_CDFs'][...]
CCPA_CDFs_99 = norm_precip(CCPA_CDFs[99, ...]) # <-- normalization
CCPA_CDFs_99[ocean_mask, :] = 0

# ------------------------------------------------------- #
# Import data
with h5py.File(camp_dir+'GFS/GEFS_OPT_MEMBERS_2021.hdf', 'r') as h5io:
    GEFS_input = h5io['apcp'][iday0:iday1, ...]

print(GEFS_input.shape)

N_days, _, EN, _, _ = GEFS_input.shape

# data pre-processing
GEFS_input[..., ocean_mask] = 0.0
GEFS_input = norm_precip(GEFS_input) # <-- normalization

# ------------------------------------------------------- #
# STEP 1
GEFS_encode = np.empty((N_days, N_leads, EN)+latent_size)

for ilead in range(N_leads):
    print('encoding lead time ind: {}'.format(ilead))
    for ien in range(EN):
        GEFS_encode[:, ilead, ien, ...] = encoder.predict(GEFS_input[:, ilead, ien, ...], verbose=0)

# ------------------------------------------------------- #
# STEP 2
GEFS_bias_correct = np.empty(GEFS_encode.shape)

for ien in range(EN):
    if ien % 5 == 0:
        print('Process ensemble member ind: {}'.format(ien))
        
    GEFS_bias_correct[:, :, ien, ...] = ViT_pred(GEFS_encode[:, :, ien, ...], model_48h, model_96h, model_144h)

# ------------------------------------------------------- #
# STEP 3
EN_LDM = int(EN * N_gen)
LDM_output = np.empty((N_days, N_leads, EN_LDM,)+latent_size)
GEFS_bias_correct_scale = rescale(GEFS_bias_correct) #<-- rescale latents

for ien in range(EN):
    if ien % 5 == 0:
        print('Process ensemble member ind: {}'.format(ien))
        
    for igen in range(N_gen):
        LDM_output[:, :, N_gen*ien+igen, ...] = LDM.generate(N_days, GEFS_bias_correct_scale[:, :, ien, ...])
        
LDM_output = scaleback(LDM_output) # scale-back latents

# ------------------------------------------------------- #
# STEP 4
OUT_VAE = np.empty((N_days, N_leads, EN_LDM)+ccpa_shape)

for ilead, lead in enumerate(LEADs):
    print('decoding lead time ind: {}'.format(ilead))
    for ien in range(EN_LDM):
        OUT_VAE[:, ilead, ien, ...] = decoder.predict(LDM_output[:, ilead, ien, ...], verbose=0)[..., 0]
        
OUT_VAE[OUT_VAE<0] = 0
OUT_VAE[..., ocean_mask] = 0

OUT_refine = np.empty((N_days, N_leads, EN_LDM)+ccpa_shape)

input_ = np.empty((N_days,)+ccpa_shape+(3,))

for ilead, lead in enumerate(LEADs):
    print('refining lead time ind: {}'.format(ilead))
    # d_ = lead // 24
    # day = d + d_
    ind_hour = lead % 24
    ind_hour = int(ind_hour/6)
    CCPA_CDFs_99_ = CCPA_CDFs_99[..., ind_hour]
    
    for ien in range(EN_LDM):
        input_[..., 0] = OUT_VAE[:, ilead, ien, ...]
        input_[..., 1] = elev_CCPA[None, ...]
        input_[..., 2] = CCPA_CDFs_99_[None, ...]
        OUT_refine[:, ilead, ien, ...] = decoder_refine.predict(input_, verbose=0)[..., 0]

OUT_refine[OUT_refine<0] = 0
OUT_refine[..., ocean_mask] = 0

# ------------------------------------------------------- #
# STEP 5
OUT_no_LDM = np.empty((N_days, N_leads, EN)+ccpa_shape)

for ilead, lead in enumerate(LEADs):
    print('decoding lead time ind: {}'.format(ilead))
    for ien in range(EN):
        OUT_no_LDM[:, ilead, ien, ...] = decoder.predict(GEFS_bias_correct[:, ilead, ien, ...], verbose=0)[..., 0]

OUT_no_LDM[OUT_no_LDM<0] = 0
OUT_no_LDM[..., ocean_mask] = 0

OUT_no_LDM_refine = np.empty((N_days, N_leads, EN)+ccpa_shape)
input_ = np.empty((N_days,)+ccpa_shape+(3,))

for ilead, lead in enumerate(LEADs):
    print('refining lead time ind: {}'.format(ilead))
    # d_ = lead // 24
    # day = d + d_
    ind_hour = lead % 24
    ind_hour = int(ind_hour/6)
    CCPA_CDFs_99_ = CCPA_CDFs_99[..., ind_hour]
    
    for ien in range(EN):
        input_[..., 0] = OUT_no_LDM[:, ilead, ien, ...]
        input_[..., 1] = elev_CCPA[None, ...]
        input_[..., 2] = CCPA_CDFs_99_[None, ...]
        OUT_no_LDM_refine[:, ilead, ien, ...] = decoder_refine.predict(input_, verbose=0)[..., 0]

OUT_no_LDM_refine[OUT_no_LDM_refine<0] = 0
OUT_no_LDM_refine[..., ocean_mask] = 0

tuple_save = (to_precip(OUT_refine), to_precip(OUT_no_LDM_refine))

label_save = ['OUT_refine', 
              'OUT_no_LDM_refine']

du.save_hdf5(tuple_save, label_save, camp_dir, 
             'LDM_results/LDM_GEFS_2021_{:03d}_{:03d}_STEP{:03d}_EN{:03d}_20240424_ATT0.hdf'.format(
                 iday0, iday1, diffusion_steps, EN_LDM))

