import os
import sys
import time
import h5py
import pygrib

import numpy as np
from glob import glob

from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('year1', help='year1')
parser.add_argument('year2', help='year2')
args = vars(parser.parse_args())

year1 = int(args['year1'])
year2 = int(args['year2'])

filenames = glob(camp_dir+'wget_GEFS_reforecast/*grib2')
ens_name = ['c00', 'p01', 'p02', 'p03', 'p04']

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    lon_CCPA = h5io['lon_CCPA'][...]
    lat_CCPA = h5io['lat_CCPA'][...]

with pygrib.open(filenames[0]) as grbio:
    lat_GFS, lon_GFS = grbio[1].latlons()
lat_GFS = lat_GFS[:360, 720:]
lon_GFS = lon_GFS[:360:, 720:]

lat_GFS = np.flipud(lat_GFS)
lon_GFS -= 360

years = np.arange(year1, year2, 1)
grb_inds = np.arange(2, 58, 2)
N_leads = len(grb_inds)
grid_shape = lon_CCPA.shape
grid_shape_gfs = lat_GFS.shape
gfs_name = camp_dir+'wget_GEFS_reforecast/apcp_sfc_{}_{}.grib2'

for year in years:
    if year % 4 == 0:
        N_days = 366
    else:
        N_days = 365
    
    GFS_save = np.zeros((N_days, N_leads,)+grid_shape)
    
    base = datetime(year, 1, 1)
    date_list = [base + timedelta(days=d) for d in range(N_days)]
    for i, dt in enumerate(date_list):
        dt_str = datetime.strftime(dt, '%Y%m%d00')
        
        ens_mean_temp = np.zeros((N_leads,)+grid_shape_gfs)
        try:
            count = 0
            for ens in ens_name:
                # Identify the *.grib2 file name
                name_ = gfs_name.format(dt_str, ens)
        
                # g
                if os.path.isfile(name_):
                    count += 1
                    with pygrib.open(name_) as grbio:
                        for ilead, ind in enumerate(grb_inds):
                            apcp_temp = grbio[int(ind)].values
                            apcp_temp = apcp_temp[:360, 720:]
                            apcp_temp = np.flipud(apcp_temp)
                            ens_mean_temp[ilead, ...] = ens_mean_temp[ilead, ...] + apcp_temp
                else:
                    print('Missing {}'.format(name_))
            if count == 0:
                print('Missing all members on {}'.format(dt_str))
                GFS_save[i, ilead, ...] = np.nan
            else:    
                ens_mean_temp = ens_mean_temp/count
                for ilead in range(N_leads):
                    lr_to_hr = RegularGridInterpolator((lat_GFS[:, 0], lon_GFS[0, :]), ens_mean_temp[ilead, ...], 
                                                       bounds_error=False, fill_value=None)
                    ens_mean_interp = lr_to_hr((lat_CCPA, lon_CCPA))
                            
                    GFS_save[i, ilead, ...] = ens_mean_interp
        except:
            print('The script failed on {}'.format(dt_str))
            GFS_save[i, ilead, ...] = np.nan
            
    tuple_save = (GFS_save,)
    label_save = ['GEFS_APCP',]
    
    du.save_hdf5(tuple_save, label_save, 
                 camp_dir+'GFS_reforecast/', 
                 'GEFS_AVE_y{}.hdf'.format(year))




