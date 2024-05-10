
import os
import sys
import time
import h5py
import numpy as np
from glob import glob
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    lon_CCPA = h5io['lon_CCPA'][...]
    lat_CCPA = h5io['lat_CCPA'][...]
    land_mask_CCPA = h5io['land_mask_CCPA'][...]

land_mask_ = land_mask_CCPA == 1.0
grid_shape = land_mask_.shape

q_bins = np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
                   0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
                   0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
                   0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
                   0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
                   0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
                   0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
                   0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
                   0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 
                   0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999])

N_bins = len(q_bins) + 1 # add max value in the end
YEARs = np.arange(2002, 2020, 1)
HOURs = np.arange(0, 24, 6)

base = datetime(2002, 1, 1)
date_list = [base + timedelta(days=day) for day in range(365*18+4)]

CCPA_name = camp_dir+'CCPA/CCPA_y{}.hdf'
CCPA_collect = np.empty((len(date_list), len(HOURs)))
quantile_save = camp_dir+'CCPA/CCPA_qbin_2002_2019_ix{:03d}_iy{:03d}_h{:02d}.npy'
q_ccpa_save = np.empty((N_bins,))

ix_inds = np.arange(0, grid_shape[0], 1).astype(int)
#ix_inds = ix_inds[::-1]
ix_inds = ix_inds[137:150]

for ix in ix_inds:
    for iy in range(grid_shape[1]):
        
        # if it is a land grid point
        if land_mask_[ix, iy]:
            
            # check if this location has been done by previous runs
            flag_not_exist = False
            for i_hour, hour in enumerate(HOURs):
                name_ = quantile_save.format(ix, iy, hour)
                if os.path.isfile(name_) is False:
                    flag_not_exist = True
                    
            # if at least one of the hours is missing --> start
            if flag_not_exist:
                print('working on ix={}, iy={}'.format(ix, iy))
                # collect all available CCPA values
                count = 0
                for i_year, year in enumerate(YEARs):
                    with h5py.File(CCPA_name.format(year), 'r') as h5io:
                        ccpa_temp = h5io['CCPA'][..., ix, iy]
                    L_temp = len(ccpa_temp)
                    CCPA_collect[count:count+L_temp, :] = ccpa_temp
                    count += L_temp
    
                for i_hour, hour in enumerate(HOURs):
                    # clear old values to be safe
                    q_ccpa_save[...] = np.nan
                    
                    # get non-NaN value collections
                    ccpa_hour = CCPA_collect[:, i_hour]
                    ccpa_hour = ccpa_hour[~np.isnan(ccpa_hour)]

                    # estimate the quantile
                    q_ccpa_save[:-1] = np.quantile(ccpa_hour, q_bins)
                    q_ccpa_save[-1] = np.max(ccpa_hour)
                    name_ = quantile_save.format(ix, iy, hour)
                    print(name_)
                    np.save(name_, q_ccpa_save)
                