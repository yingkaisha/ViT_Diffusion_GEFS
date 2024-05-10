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

hours = ['00', '06', '12', '18']
filename = camp_dir+'wget_CCPA/ccpa.{}/{}/*06h*'

for year in range(2002, 2024, 1):
    if year % 4 == 0:
        N_days = 366
    else:
        N_days = 365

    APCP = np.empty((N_days, 4,)+lon_CCPA.shape)
    APCP[...] = np.nan
    
    base = datetime(year, 1, 1)
    date_list = [base + timedelta(days=d) for d in range(N_days)]
    
    for d, dt in enumerate(date_list):
        for ih, h in enumerate(hours):
            dt_str = datetime.strftime(dt, '%Y%m%d')
            filename_ = glob(filename.format(dt_str, h))
    
            if len(filename_) > 0:
                with pygrib.open(filename_[0]) as grbio:
                    apcp = grbio[1].values
                    apcp = np.array(apcp)
                    apcp[apcp>1000] = 0.0
                    
                APCP[d, ih, ...] = apcp
    
    tuple_save = (APCP,)
    label_save = ['CCPA',]
    du.save_hdf5(tuple_save, label_save, camp_dir+'CCPA/', 'CCPA_y{}.hdf'.format(year))
