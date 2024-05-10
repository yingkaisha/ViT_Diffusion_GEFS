# ------------------------------------------------------- #
# system tools
import sys
import time
from glob import glob
from datetime import datetime, timedelta

# ------------------------------------------------------- #
# data tools
import h5py
import pygrib
import numpy as np
import netCDF4 as nc

# ------------------------------------------------------- #
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

import united_states

def str_search(strs, keys, verbose=True):
    '''
    Return the index of each keys element from strs
    e.g.
        strs = ['a', 'b', 'c', 'd']; keys = ['a', 'c']
        str_serach(...) --> [0, 2]
    '''
    ind = []
    for key in keys:
        ind_temp = [i for i,s in enumerate(strs) if key in s]
        if len(ind_temp) == 1:
            ind.append(ind_temp[0])
        elif len(ind_temp) > 1:
            if verbose:
                print('duplicate items (will pick the last one):')
            for ind_d in ind_temp:
                if verbose:
                    print('{} --> {}'.format(ind_d, strs[ind_d]))
            ind.append(ind_d)
        else:
            if verbose:
                print('item {} not found.'.format(key))
            ind.append(9999)
    return ind

state_names = [
    'Alabama',
    'Arizona',
    'Arkansas',
    'California',
    'Colorado',
    'Connecticut',
    'Delaware',
    'Florida',
    'Georgia',
    'Idaho',
    'Illinois',
    'Indiana',
    'Iowa',
    'Kansas',
    'Kentucky',
    'Louisiana',
    'Maine',
    'Maryland',
    'Massachusetts',
    'Michigan',
    'Minnesota',
    'Mississippi',
    'Missouri',
    'Montana',
    'Nebraska',
    'Nevada',
    'New Hampshire',
    'New Jersey',
    'New Mexico',
    'New York',
    'North Carolina',
    'North Dakota',
    'Ohio',
    'Oklahoma',
    'Oregon',
    'Pennsylvania',
    'Rhode Island',
    'South Carolina',
    'South Dakota',
    'Tennessee',
    'Texas',
    'Utah',
    'Vermont',
    'Virginia',
    'Washington',
    'West Virginia',
    'Wisconsin',
    'Wyoming']

us = united_states.UnitedStates()

state_names_match = []

for name in state_names:
    state_names_match.append(name.replace(" ", "_"))

with h5py.File(save_dir+'CCPA_domain.hdf', 'r') as h5io:
    lon_CCPA = h5io['lon_CCPA'][...]
    lat_CCPA = h5io['lat_CCPA'][...]
    elev_CCPA = h5io['elev_CCPA'][...]
    land_mask_CCPA = h5io['land_mask_CCPA'][...]

shape_ccpa = lon_CCPA.shape
state_id = np.empty(shape_ccpa)
state_id[...] = np.nan

for i in range(shape_ccpa[0]):
    for j in range(shape_ccpa[1]):
        result = us.from_coords(lat_CCPA[i, j], lon_CCPA[i, j])
        if len(result) > 0:
            name = result[0].name
            name = name.replace(" ", "_")
            if name == 'Virginia':
                state_id[i, j] = 43
            elif name == 'West_Virginia':
                state_id[i, j] = 45
            else:
                state_id[i, j] = str_search(state_names_match, [name,])[0]

tuple_save = (lon_CCPA, lat_CCPA, elev_CCPA, land_mask_CCPA, state_id)
label_save = ['lon_CCPA', 'lat_CCPA', 'elev_CCPA', 'land_mask_CCPA', 'state_id']
du.save_hdf5(tuple_save, label_save, save_dir, 'CCPA_domain_backup.hdf')