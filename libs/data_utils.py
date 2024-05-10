# general tools
# import sys
# from glob import glob
# from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
# import pandas as pd

# stats tools
# from scipy.spatial import cKDTree
# from scipy.interpolate import interp2d
# from scipy.interpolate import NearestNDInterpolator
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
from scipy.stats import gaussian_kde

def kde_estimator(X_coords, Y_coords, N=200):
    XY_coords = np.concatenate((X_coords[..., None], Y_coords[..., None]), axis=1)
    kde = gaussian_kde(XY_coords.T)

    X_flat = np.linspace(np.min(X_coords), np.max(X_coords), N)
    Y_flat = np.linspace(np.min(Y_coords), np.max(Y_coords), N)

    X_grids, Y_grids = np.meshgrid(X_flat, Y_flat)
    grid_coords = np.append(X_grids.reshape(-1,1), Y_grids.reshape(-1,1), axis=1)
    OUT = kde(grid_coords.T)
    return X_grids, Y_grids, OUT.reshape(N, N)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def shuffle_ind(L):
    '''
    shuffle indices
    L: length of dimension
    '''
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind

def is_leap_year(year):
    '''
    Determine whether a year is a leap year.
    '''
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def save_hdf5(p_group, labels, out_dir, filename='example.hdf'):
    '''
    Save data into a signle hdf5
        - p_group: datasets combined in one tuple;
        - labels: list of strings;
        - out_dir: output path;
        - filename: example.hdf;
    **label has initial 'x' means ENCODED strings
    '''    
    name = out_dir+filename
    hdf = h5py.File(name, 'w')
    for i, label in enumerate(labels):
        if label[0] != 'x':
            hdf.create_dataset(label, data=p_group[i])
        else:
            string = p_group[i]
            hdf.create_dataset(label, (len(string), 1), 'S10', string)
    hdf.close()
    print('Save to {}'.format(name))
