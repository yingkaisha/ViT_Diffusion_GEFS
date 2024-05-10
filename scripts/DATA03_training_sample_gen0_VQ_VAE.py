'''
Generate training samples for VQ-VAE
'''

import os
import sys
import time
import h5py
import pygrib
from glob import glob
import numpy as np
from datetime import datetime, timedelta

# ------------------------------------------------------- #
# Import customized modules and settings
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

BATCH_dir = camp_dir+'BATCH_CCPA_full/'
batch_name = 'CCPA_{}_{}.npy' #.format(datetime, hour-of-day)
grid_shape = (224, 464)
hours = ['00', '06', '12', '18']

# ------------------------------------------------------- #
# Datetime information

base = datetime(2002, 1, 1)
date_list = [base + timedelta(days=d) for d in range(365*22+5)]

# Pick CCPA 06h products
filename = camp_dir+'wget_CCPA/ccpa.{}/{}/*06h*'

# ------------------------------------------------------- #
# The main sample generation loop
for d, dt in enumerate(date_list):
    for h in hours:
        dt_str = datetime.strftime(dt, '%Y%m%d')
        filename_ = glob(filename.format(dt_str, h))
        
        if len(filename_) > 0:
            #print(filename_)
            with pygrib.open(filename_[0]) as grbio:
                apcp = grbio[1].values
                apcp = np.array(apcp)
                apcp[apcp>1000] = 0.0
                apcp[land_mask_CCPA==0] = 0
                
            apcp = np.log(0.1*apcp+1)
            save_name = BATCH_dir+batch_name.format(dt_str, h)
            #print(save_name)
            np.save(save_name, apcp)

