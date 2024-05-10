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

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    lon_CCPA = h5io['lon_CCPA'][...]
    lat_CCPA = h5io['lat_CCPA'][...]
    land_mask_CCPA = h5io['land_mask_CCPA'][...]

land_mask_CCPA = land_mask_CCPA == 1.0
grid_shape = land_mask_CCPA.shape

LEADs = np.arange(6, 144+6, 6)
N_leads = len(LEADs)
N_days = 365

with h5py.File(camp_dir+'CCPA/CCPA_lead_y{}.hdf'.format(2021), 'r') as h5io:
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
    with h5py.File(camp_dir+'GFS/GEFS_OPT_MEMBERS_2021.hdf', 'r') as h5io:
        apcp = h5io['apcp'][...]
    PRED = apcp

else:
    print('Unknown data')
    raise

CRPS_grids = np.empty((N_days, N_leads)+grid_shape)
for ilead, lead in enumerate(LEADs):
    crps_ilead, _, _ = vu.CRPS_2d(CCPA_true[:, ilead, ...], PRED[:, ilead, ...], land_mask=land_mask_CCPA)
    CRPS_grids[:, ilead, ...] = crps_ilead

np.save(result_dir+'CRPS_{}.npy'.format(method), CRPS_grids)










