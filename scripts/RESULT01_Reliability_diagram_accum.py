# general tools
import os
import sys
import time
import h5py
import zarr
import numba as nb
from glob import glob

import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import verif_utils as vu

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('method_name', help='method_name')
args = vars(parser.parse_args())

method = args['method_name']

N_boost = 100
hist_bins = np.linspace(0, 1, 14)
prefix = result_dir+'{}_reliability_accum.npy'

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    lon_CCPA = h5io['lon_CCPA'][...]
    lat_CCPA = h5io['lat_CCPA'][...]
    land_mask_CCPA = h5io['land_mask_CCPA'][...]

land_mask_CCPA = land_mask_CCPA == 1.0
grid_shape = land_mask_CCPA.shape

q_bins = np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
                   0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
                   0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
                   0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
                   0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
                   0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
                   0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
                   0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
                   0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 
                   0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1.0])

with h5py.File(camp_dir+'CCPA/CCPA_CDFs_lead_2002_2019.hdf', 'r') as h5io:
    CCPA_CDFs = h5io['CCPA_CDFs_lead'][...]

LEADs = np.arange(6, 144+6, 6)
N_leads = len(LEADs)
N_days = 365

with h5py.File(camp_dir+'CCPA/CCPA_lead_y2021.hdf', 'r') as h5io:
    CCPA_true = h5io['CCPA_lead'][...]

if method == 'ViT':
    filename = result_dir+'VIT_FULL_GEFS_2021_STEP100_EN031_20240420_ATT0.zarr'
    PRED = zarr.open(filename, 'r')
    PRED = np.array(PRED)

elif method == 'LDM':
    filename = result_dir+'LDM_FULL_GEFS_2021_STEP100_EN062_20240420_ATT0.hdf'
    with h5py.File(filename, 'r') as h5io:
        PRED = h5io['LDM_FULL'][...]
        
elif method == 'AnEn':
    EN = 31
    AnEn_name = camp_dir+'AnEn_baseline/AnEn_ECC_2021_lead{:02d}_.hdf'
    AnEn = np.empty((365, N_leads, EN) + grid_shape); AnEn[...] = np.nan
    for ilead, lead in enumerate(LEADs):
        with h5py.File(AnEn_name.format(lead), 'r') as h5io:
            temp = h5io['AnEn'][:, :EN, ...]
        AnEn[:, ilead, ...] = temp
    PRED = AnEn
    
elif method == 'RAW':
    filename = camp_dir+'GFS/GEFS_OPT_MEMBERS_2021.hdf'
    with h5py.File(filename, 'r') as h5io:
        apcp = h5io['apcp'][...]
    PRED = apcp

else:
    print('Unknown data')
    raise

CCPA_CDFs_accum = np.mean(CCPA_CDFs, axis=1)

def reliability_quantile_thres(CCPA_true, LDM, land_mask_CCPA, thres, Q, N_boost, hist_bins, prefix, method):
    Q_ = CCPA_CDFs[thres, ...]
    
    LDM_flag = LDM[:, ...] > Q_[None, :, None, ...]
    LDM_flag = np.nanmean(LDM_flag, axis=2)
    
    LDM_flag = LDM_flag[..., land_mask_CCPA]
    
    CCPA_true_flag = CCPA_true > Q_[None, ...]
    CCPA_true_flag = CCPA_true_flag[..., land_mask_CCPA]
    
    y_true = CCPA_true_flag.ravel()
    y_pred = LDM_flag.ravel()
    
    output_bundle = vu.reliability_diagram_bootstrap(y_true, y_pred, N_boost=N_boost, hist_bins=hist_bins)
    prob_true, prob_pred, hist_bins_, use_, o_bar, prob_pred_mean, prob_true_mean = output_bundle
    
    rel, res, o_bar, bs = vu.bss_component_calc(prob_pred_mean, prob_true_mean, o_bar, use_)
    save_bundle = output_bundle + (rel, res, o_bar, bs)
    
    save_name = prefix.format(method)
    save_dict = {}
    save_dict['save_bundle'] = save_bundle
    np.save(save_name, save_dict)
    print(save_name)

reliability_quantile_thres(CCPA_true, PRED, land_mask_CCPA, 99, CCPA_CDFs_accum, 100, hist_bins, prefix, method)


