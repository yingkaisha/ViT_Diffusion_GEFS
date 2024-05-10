import os
import sys
import time
import numpy as np
from glob import glob

# ------------------------------------------------------- #
# Turn-off warnings
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ------------------------------------------------------- #
# Turn-off tensoflow-specific warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

# ------------------------------------------------------- #
# Import customized modules and settings
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

# ------------------------------------------------------- #
# Hyperparameters
filter_nums = [64, 128] # number of convolution kernels per down-/upsampling layer 
latent_dim = 4 # number of latent feature channels
activation = 'gelu' # activation function
num_embeddings = 128 #128 # number of the VQ codes

input_size = (224, 464, 1) # size of MRMS input
latent_size = (14, 29, latent_dim) # size of compressed latent features

load_weights = True

model_name_load = model_dir+'baseline/VQ_VAE_stack1_pretrain/'
model_name_save = model_dir+'baseline/VQ_VAE_stack1_pretrain/'

# separate save encoder and decoder
model_name_encoder_save = model_dir+'models/VQ_VAE_encoder_stack1_tune0'
model_name_decoder_save = model_dir+'models/VQ_VAE_decoder_stack1_tune0'


drop_encode = True
drop_decode = True

lr = 1e-5 # learning rate
# samples per epoch = N_batch * batch_size
epochs = 99999
N_batch = 4
batch_size = 32*16
batch_size_ = 32

# ------------------------------------------------------- #
# Validation set prep
# location of training data
BATCH_dir = camp_dir+'BATCH_CCPA_full/'
# validation set size

# collect validation set sampales
filenames = sorted(glob(BATCH_dir+'*.npy'))
filenames = filenames[:-3648]
L = len(filenames)

filename_valid = filenames[::8][:2627]
#filename_valid = filenames[::8][:500] # samller validation set size
L_valid = len(filename_valid)

Y_valid = np.empty((L_valid, 224, 464, 1))
Y_valid[...] = np.nan

for i, name in enumerate(filename_valid):
    Y_valid[i, ..., 0] = np.load(name)
    
# # make sure the validation set contains no NaNs
# print('NaN grids = {}'.format(np.sum(np.isnan(Y_valid))))

# ------------------------------------------------------- #
# Training data prep
# Capture all the training set
filenames = sorted(glob(BATCH_dir+'*.npy'))
filenames = filenames[:-3648]
filename_train = list(set(filenames) - set(filename_valid))

L_train = len(filename_train)

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

# ------------------------------------------------------- #
# Distributed training

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    # ------------------------------------------------------- #
    # VQ-VAE
    # ---------------- encoder ----------------- #
    encoder = mu.VQ_VAE_encoder(input_size, filter_nums, latent_dim, num_embeddings, activation, drop_encode)
    decoder = mu.VQ_VAE_decoder(latent_size, filter_nums, activation, drop_decode)

    # Connect the encoder and decoder
    X = keras.Input(shape=input_size)
    X_encode = encoder(X)
    X_decode = decoder(X_encode)
    model_vqvae = keras.Model(X, X_decode)
    
    # subclass to VAE training
    vqvae_trainer = mu.VQVAETrainer(model_vqvae, 1.0, latent_dim, num_embeddings)
    
    # load weights
    if load_weights:
        W_old = mu.dummy_loader(model_name_load)
        vqvae_trainer.vqvae.set_weights(W_old)
    
    # compile
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    # ----------------------------------------------- #
    # Major training loop + training batch generation
    
    Y_batch = np.empty((batch_size, 224, 464, 1))
    Y_batch[...] = np.nan
    
    for i in range(epochs):
        
        print('epoch = {}'.format(i))
        if i == 0:
            model_ = vqvae_trainer.vqvae
            Y_pred = model_.predict(Y_valid)
            Y_pred[Y_pred<0] = 0
            record = du.mean_absolute_error(Y_valid, Y_pred)
            print('Initial validation loss: {}'.format(record))
        
        start_time = time.time()
        for j in range(N_batch):
            
            inds_rnd = du.shuffle_ind(L_train)
            inds_ = inds_rnd[:batch_size]
    
            for k, ind in enumerate(inds_):
                # import batch data
                name = filename_train[ind]
                Y_batch[k, ..., 0] = np.load(name)

            Y_batch += np.random.normal(0, 0.01, size=Y_batch.shape)
            vqvae_trainer.fit(Y_batch, epochs=1, batch_size=batch_size_, verbose=0)
            
        # on epoch-end
        model_ = vqvae_trainer.vqvae
        Y_pred = model_.predict(Y_valid)
        Y_pred[Y_pred<0] = 0
        record_temp = du.mean_absolute_error(Y_valid, Y_pred)
    
        if record - record_temp > min_del:
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            model_ = vqvae_trainer.vqvae
            print("Save to {}".format(model_name_save))
            model_.save(model_name_save)
            
        else:
            print('Validation loss {} NOT improved'.format(record_temp))
        
        print("--- %s seconds ---" % (time.time() - start_time))
        # mannual callbacks


