# sys tools
import sys
import time

# data tools
import h5py
import numpy as np
import numba as nb

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import analog_utils as au
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead1', help='lead1')
parser.add_argument('lead2', help='lead2')
args = vars(parser.parse_args())

lead1 = int(args['lead1'])
lead2 = int(args['lead2'])

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    lon_CCPA = h5io['lon_CCPA'][...]
    lat_CCPA = h5io['lat_CCPA'][...]
    land_mask_CCPA = h5io['land_mask_CCPA'][...]

land_mask_ = land_mask_CCPA == 1.0
grid_shape = land_mask_.shape
grid_num = np.sum(land_mask_)

# ================================================ #
# AnEn parameters
year_fcst = 2021

day0 = 0
day1 = 365
L_fcst_days = day1 - day0

EN = 50
year_analog = np.arange(2002, 2019)
LEADs = np.arange(6, 144+6, 6)

# ================================================ #

AnEn_grid = np.empty((L_fcst_days, EN)+grid_shape)

for lead, lead_val in enumerate(LEADs):
    if lead_val >= lead1 and lead_val <= lead2:
        print("Processing lead time = {}".format(lead_val))
        # ------------------------------------------------- #
        # Import reforecast
        APCP = ()
        for year in year_analog:
            with h5py.File(camp_dir+'GFS_reforecast/GEFS_AVE_y{}.hdf'.format(year), 'r') as h5io:
                apcp_temp = h5io['GEFS_APCP'][:, lead, ...][:, land_mask_]
                
            # these days have NaN, let AnEN ignore
            if year == 2002:
                apcp_temp[:2, ...] = 9999
                
            APCP += (apcp_temp,)
        
        # # ------------------------------------------------- #
        # Import CCPA
        CCPA = ()
        for year in year_analog:
            with h5py.File(camp_dir+'CCPA/CCPA_lead_y{}.hdf'.format(year), 'r') as h5io:
                ccpa_temp = h5io['CCPA_lead'][:, lead, ...][:, land_mask_]
            CCPA += (ccpa_temp,)
            
        # ------------------------------------------------- #
        # importing new fcst
        with h5py.File(camp_dir+'GFS/geave_y{}.hdf'.format(year_fcst), 'r') as h5io:
            fcst_apcp = h5io['apcp'][:, lead, ...][:, land_mask_]
    
        AnEn = au.analog_search(day0, day1, year_analog, fcst_apcp, APCP, CCPA)
        
        AnEn_grid[...] = np.nan
        
        for i in range(L_fcst_days):
            for j in range(EN):
                AnEn_grid[i, j, land_mask_] = AnEn[i, ..., j]
    
        tuple_save = (AnEn_grid,)
        label_save = ['AnEn',]
        du.save_hdf5(tuple_save, label_save, camp_dir+'AnEn_baseline/', 
                     'AnEn_ECC_{}_lead{:02d}_.hdf'.format(year_fcst, lead_val))
