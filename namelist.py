import numpy as np

NCAR_dir = '$USER/NCAR/'

work_dir = '$USER/work/'
camp_dir = '$USER/campaign/'
result_dir = '$USER/LDM_results/'
save_dir = data_dir = model_dir = '$USER/GAN/'

# figure storage
fig_dir = '$USER/figures/'

# Matplotlib figure export settings
fig_keys = {'dpi':250, 
            'orientation':'portrait', 
            'bbox_inches':'tight', 
            'pad_inches':0.1, 
            'transparent':False}

# colors
#
rgb_array = np.array([[0.85      , 0.85      , 0.85      , 1.        ],
                      [0.66666667, 1.        , 1.        , 1.        ],
                      [0.33333333, 0.62745098, 1.        , 1.        ],
                      [0.11372549, 0.        , 1.        , 1.        ],
                      [0.37647059, 0.81176471, 0.56862745, 1.        ],
                      [0.10196078, 0.59607843, 0.31372549, 1.        ],
                      [0.56862745, 0.81176471, 0.37647059, 1.        ],
                      [0.85098039, 0.9372549 , 0.54509804, 1.        ],
                      [1.        , 1.        , 0.4       , 1.        ],
                      [1.        , 0.8       , 0.4       , 1.        ],
                      [1.        , 0.53333333, 0.29803922, 1.        ],
                      [1.        , 0.09803922, 0.09803922, 1.        ],
                      [0.8       , 0.23921569, 0.23921569, 1.        ],
                      [0.64705882, 0.19215686, 0.19215686, 1.        ],
                      [0.55      , 0.        , 0.        , 1.        ]])
    
blue   = rgb_array[3, :]  # blue
cyan   = rgb_array[2, :]  # cyan
lgreen = rgb_array[4, :]  # light green
green  = rgb_array[5, :]  # dark green
yellow = rgb_array[8, :]  # yellow
orange = rgb_array[-6, :] # orange
red    = rgb_array[-3, :] # red
