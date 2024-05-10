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

thres_list = [5, 10, 15, 20, 25, 30, 35]
N_boost = 100
hist_bins = np.linspace(0, 1, 14)
prefix = result_dir+'{}_reliability_thres{}_lead{}_{}.npy'

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    lon_CCPA = h5io['lon_CCPA'][...]
    lat_CCPA = h5io['lat_CCPA'][...]
    land_mask_CCPA = h5io['land_mask_CCPA'][...]

land_mask_CCPA = land_mask_CCPA == 1.0
grid_shape = land_mask_CCPA.shape

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

def reliability_domain_thres(CCPA_true, LDM, land_mask_CCPA, thres, lead_ind_range, N_boost, hist_bins, prefix, method):
    CCPA_true_flag = CCPA_true[:, lead_ind_range[0]:lead_ind_range[1], ...] > thres
    CCPA_true_flag = CCPA_true_flag[..., land_mask_CCPA]
    
    LDM_flag = vu.ens_to_prob(LDM, thres, lead_ind_range)
    LDM_flag = np.nanmean(LDM_flag, axis=2)
    LDM_flag = LDM_flag[..., land_mask_CCPA]

    y_true = CCPA_true_flag.ravel()
    y_pred = LDM_flag.ravel()
    
    output_bundle = vu.reliability_diagram_bootstrap(y_true, y_pred, N_boost=N_boost, hist_bins=hist_bins)
    prob_true, prob_pred, hist_bins_, use_, o_bar, prob_pred_mean, prob_true_mean = output_bundle

    rel, res, o_bar, bs = vu.bss_component_calc(prob_pred_mean, prob_true_mean, o_bar, use_)
    save_bundle = output_bundle + (rel, res, o_bar, bs)
    
    save_name = prefix.format(method, thres, lead_ind_range[0], lead_ind_range[1])
    save_dict = {}
    save_dict['save_bundle'] = save_bundle
    np.save(save_name, save_dict)
    print(save_name)


for thres in thres_list:
    lead_ind_range = [0, 8]
    reliability_domain_thres(CCPA_true, PRED, land_mask_CCPA, thres, 
                             lead_ind_range, N_boost, hist_bins, prefix, method)
    lead_ind_range = [8, 16]
    reliability_domain_thres(CCPA_true, PRED, land_mask_CCPA, thres, 
                             lead_ind_range, N_boost, hist_bins, prefix, method)
    lead_ind_range = [16, 24]
    reliability_domain_thres(CCPA_true, PRED, land_mask_CCPA, thres, 
                             lead_ind_range, N_boost, hist_bins, prefix, method)


