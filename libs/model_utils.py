import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================================================= #
# Helper functions
def dummy_loader(model_path):
    '''
    Load weights from keras v2 model file.
    '''
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

def set_seeds(seed):
    '''
    Numpy and Tensorflow random seed
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def padding_3d(size_before, size_after):
    '''
    (x) Not used
    Given before and after tensor sizes, it returns the size that needs to be padded.
    Proposed for np.pad().
    '''
    pad_dim0_left = (size_after[0] - size_before[0]) // 2
    pad_dim1_left = (size_after[1] - size_before[1]) // 2
    pad_dim2_left = (size_after[2] - size_before[2]) // 2

    pad_dim0_right = (size_after[0] - size_before[0]) - pad_dim0_left
    pad_dim1_right = (size_after[1] - size_before[1]) - pad_dim1_left
    pad_dim2_right = (size_after[2] - size_before[2]) - pad_dim2_left
    
    return ((pad_dim0_left, pad_dim0_right), 
            (pad_dim1_left, pad_dim1_right), 
            (pad_dim2_left, pad_dim2_right))

def cosine_schedule(num_timesteps, l_min, l_max):
    '''
    The function that updates learning rates based on cosine annealing schedule.
    '''
    return l_min + 0.5*(l_max - l_min)*(1+np.cos(np.arange(num_timesteps)/num_timesteps*np.pi))

def encode_mapping(data, encoder, latent_size):
    '''
    Run encoder within for loops to convert gridded forecasts into encoded information.
    '''
    data_shape = data.shape
    
    if len(data_shape) == 4:
        N_leads, EN, _, _ = data_shape
    elif len(data_shape) == 3:
        N_leads, _, _ = data_shape
        EN = 1
        data = data[:, None, ...]
        
    data_encode = np.empty((N_leads, EN)+latent_size)
    
    for ilead in range(N_leads):
        for ien in range(EN):
           data_encode[ilead, ien, ...] = encoder.predict(data[ilead, ien, ...][None, ...], verbose=0)
            
    return data_encode

# ================================================================= #
# VQ-VAE section

class VectorQuantizer(layers.Layer):
    '''
    VQ-layer with commitment loss and codebook loss.
    '''
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True, name="embeddings_vqvae",)

    def call(self, x):
        # Calculate the shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Loss computations
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                     + tf.reduce_sum(self.embeddings ** 2, axis=0)
                     - 2 * similarity)

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

class VQVAETrainer(keras.models.Model):
    '''
    VQ-VAE warpper on Keras v2 model.
    '''
    def __init__(self, model, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = model

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        
        # reconstruction loss
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        
        # commitment loss and codebook loss from the VQ-layer
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.vq_loss_tracker,]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return { "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "vqvae_loss": self.vq_loss_tracker.result(),}

def resblock_vqvae(X, kernel_size, filter_num, activation):
    '''
    Residual block designs for VQ-VAE.
    Conv --> BN --> GELU --> Conv --> (skip) --> BN --> GELU
    '''
    Fx = layers.Conv2D(filter_num, kernel_size, padding='same')(X)
    Fx = layers.BatchNormalization()(Fx)
    Fx = layers.Activation(activation)(Fx)
    Fx = layers.Conv2D(filter_num, kernel_size, padding='same')(Fx)
    out = layers.Add()([X, Fx])
    out = layers.BatchNormalization(axis=-1)(out)
    out = layers.Activation(activation)(out)
    return out

def VQ_VAE_encoder(input_size, filter_nums, latent_dim, num_embeddings, activation, drop_encode, stride=4, stack=1):
    '''
    VQ-VAE encoder
    '''
    # Input layer
    encoder_in = keras.Input(shape=input_size)
    X = encoder_in

    # ------------------------------------------------------------------------------ #
    # Blocks = Conv2D - BN - GELU - Resblock - Conv2D(stride=4) - BN - GELU - dropout
    # ------------------------------------------------------------------------------ #
    
    # Block 1
    X = layers.Conv2D(filter_nums[0], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)

    for i in range(stack):
        X = resblock_vqvae(X, 3, filter_nums[0], activation)
    
    X = layers.Conv2D(filter_nums[0], stride, strides=stride, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)
    
    if drop_encode:
        X = layers.Dropout(rate=0.1)(X)
    
    # block2
    X = layers.Conv2D(filter_nums[1], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)

    for i in range(stack):
        X = resblock_vqvae(X, 3, filter_nums[1], activation)
    
    X = layers.Conv2D(filter_nums[1], stride, strides=stride, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)
    
    if drop_encode:
        X = layers.Dropout(rate=0.1)(X)

    # output conv stacks
    X = layers.Conv2D(filter_nums[1], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)
    
    encoder_out = layers.Conv2D(latent_dim, 1, padding="same")(X)
    
    # # --- VQ layer config --- #
    vq_layer = VectorQuantizer(num_embeddings, latent_dim)
    X_VQ = vq_layer(encoder_out)
    # # --- VQ layer config --- #

    # model
    model_encoder = keras.Model(encoder_in, X_VQ)
    return model_encoder

def VQ_VAE_decoder(latent_size, filter_nums, activation, drop_decode, stride=4, stack=1):
    '''
    VQ-VAE decoder
    '''
    # Input layer
    decoder_in = keras.Input(shape=latent_size)
    X = decoder_in
    
    # Initial conv
    X = layers.Conv2D(filter_nums[1], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)

    # ----------------------------------------------------------- #
    # Upsample2D(size=4) - Conv2D - BN - GELU - Resblock - dropout
    # ----------------------------------------------------------- #
    
    # Block 1
    X = layers.UpSampling2D(size=(stride, stride), interpolation="bilinear")(X)
    
    X = layers.Conv2D(filter_nums[1], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)

    for i in range(stack):
        X = resblock_vqvae(X, 3, filter_nums[1], activation)
    
    if drop_decode:
        X = layers.Dropout(rate=0.1)(X)

    # Block 2
    X = layers.UpSampling2D(size=(stride, stride), interpolation="bilinear")(X)
    
    X = layers.Conv2D(filter_nums[0], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)

    for i in range(stack):
        X = resblock_vqvae(X, 3, filter_nums[0], activation)
        
    # Output conv stacks
    X = layers.Conv2D(filter_nums[0], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation)(X)
    
    decoder_out = layers.Conv2D(1, 1, padding="same")(X)

    # model
    model_decoder = keras.Model(decoder_in, decoder_out)

    return model_decoder

# blocks that connects to the decoder of VQ-VAE backboen
def VQ_VAE_refine_blocks(input_size, filter_nums):

    IN = layers.Input(input_size)
    X = IN
    
    X = layers.Conv2D(filter_nums[0], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('gelu')(X)

    X = layers.Conv2D(filter_nums[1], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('gelu')(X)
    
    X = resblock_vqvae(X, kernel_size=3, filter_num=filter_nums[1], activation='gelu')
    X = resblock_vqvae(X, kernel_size=3, filter_num=filter_nums[1], activation='gelu')
    
    X = layers.Conv2D(filter_nums[2], 3, padding="same")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('gelu')(X)
    
    OUT = layers.Conv2D(1, 1, padding="same")(X)
    
    return keras.Model(IN, OUT)

# ================================================================= #
# 3d bias-correction ViT section

class TubeletEmbedding(layers.Layer):
    '''
    Cube Embedding using Conv3D.
    '''
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(filters=embed_dim, kernel_size=patch_size, 
                                        strides=patch_size, padding="VALID",)
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    '''
    Positional Embeddings on flattened patches using keras.layers.Embedding()
    '''
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(input_dim=num_tokens, output_dim=self.embed_dim)
        self.positions = tf.range(0, num_tokens, 1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
        
def ViT3d_corrector(input_size, output_size, patch_size, project_dim, N_layers, N_heads):
    '''
    3-D ViT for bias correction.
    '''
    # Compute the number of patches on each dimension
    N_patches = (input_size[0] // patch_size[0], input_size[1] // patch_size[1], input_size[2] // patch_size[2])

    # Define layer: Tubelet embedding with Conv3D
    tubelet_embedder = TubeletEmbedding(embed_dim=project_dim, patch_size=patch_size)

    # Define layer: Positional encoder with keras built-in Embedding
    positional_encoder = PositionalEncoder(embed_dim=project_dim)

    # 3d inputs of (lead_time, lat, lon)
    inputs = layers.Input(input_size)
    # Subset inputs into patches
    patches = tubelet_embedder(inputs)
    # Encode positional information
    encoded_patches = positional_encoder(patches)

    # ViT stracks
    # ============================================================================================== #
    # Layer Norm - MH Attention - skip connection - Layer Norm - Dense layer stacks - skip connection
    # ============================================================================================== #
    for n_layer in range(N_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=N_heads, 
                                                     key_dim=project_dim // N_heads, 
                                                     dropout=0.1)(x1, x1)
        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])
    
        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential([layers.Dense(units=project_dim * 4),
                               layers.Activation('gelu'),
                               layers.Dense(units=project_dim),
                               layers.Activation('gelu'),])(x3)
        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
        if n_layer >= 1 and n_layer < N_layers-1:
            encoded_patches = layers.Dropout(rate=0.1)(encoded_patches)
    
    # Revert Tuple Embedding with dense layer
    ## Reshape to (Batch, Time, Lat, Lon, Proj_dim)
    x_test = tf.reshape(encoded_patches, (-1, N_patches[0], N_patches[1], N_patches[2], project_dim))
    
    ## Expand Proj_dim to patch_size * channels
    x_test = layers.Dense(output_size[-1]*patch_size[0]*patch_size[1]*patch_size[2])(x_test)
    
    ## Recover to the output size with reshape and permute
    x_test = tf.reshape(x_test, (-1, N_patches[0], N_patches[1], N_patches[2], 
                                 patch_size[0], patch_size[1], patch_size[2], output_size[-1]))
    
    x_test = tf.transpose(x_test, perm=[0, 1, 4, 2, 5, 3, 6, 7])
    
    x_test = tf.reshape(x_test, (-1, N_patches[0]*patch_size[0], 
                                 N_patches[1]*patch_size[1], 
                                 N_patches[2]*patch_size[2], output_size[-1]))
    # model
    model = keras.Model(inputs=inputs, outputs=x_test)
    
    return model

# ================================================================= #
# 3D diffusion model

def sinusoidal_embedding(x, embedding_dims):
    '''
    Embed diffusion step information using sine and cosine functions.
    '''
    # no more than 1000 diffusion steps
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    
    frequencies = tf.math.exp(
        tf.linspace(tf.math.log(embedding_min_frequency), 
                    tf.math.log(embedding_max_frequency), embedding_dims // 2))
    
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat([tf.math.sin(angular_speeds * x), 
                            tf.math.cos(angular_speeds * x)], axis=-1)
    return embeddings


def ResidualBlock(width, activation="swish"):
    '''
    The Residual block design of 3-D diffusion model.
    Ref. to Res-Unet.
    '''
    def apply(x):
        # match input shapes
        input_width = x.shape[-1]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv3D(width, kernel_size=1)(x)
        # BN - Conv3D - Conv3D - skip connect
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(width, kernel_size=3, padding="same", activation=activation)(x)
        x = layers.Conv3D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply


def DownBlock(width, block_depth):
    # Downsampling removed
    def apply(x):
        x, skips = x
        # save the ouput of each block depth for concatenation
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        return x
    return apply


def UpBlock(width, block_depth):
    # Upsampling removed
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply


def get_network(input_size, input_condition_size, output_size, embedding_dims, widths, block_depth):
    '''
    3-D diffusion model builder.
    '''
    # noise input
    noisy_images = keras.Input(shape=input_size)
    # conditional input
    conditions = keras.Input(shape=input_condition_size)
    # noise variance embeddings
    noise_variances = keras.Input(shape=(1, 1, 1, 1))
    e = layers.Lambda(sinusoidal_embedding, 
                      arguments={'embedding_dims':embedding_dims}, 
                      output_shape=(1, 1, 1, 32))(noise_variances)
    # expand noise variance to the (time, lat, lon) sizes
    e = layers.UpSampling3D(size=(input_size[0], input_size[1], input_size[2]))(e)

    # initial conv (noise)
    x = layers.Conv3D(widths[0], kernel_size=1)(noisy_images)
    # initial conv (conditional input)
    x_condition = layers.Conv3D(widths[0], kernel_size=1)(conditions)
    # concatenate noise and conditional inputs
    x = layers.Concatenate()([x, x_condition, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    # output layer
    x = layers.Conv3D(output_size[-1], kernel_size=1, kernel_initializer="zeros")(x)
    
    return keras.Model([noisy_images, conditions, noise_variances], x, name="residual_unet")

class DiffusionModel(keras.Model):
    '''
    Diffusion model
    '''
    def __init__(self, input_size, input_condition_size, output_size, 
                 diffusion_steps, min_signal_rate, max_signal_rate, 
                 embedding_dims, widths, block_depth, ema):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        # ----- #
        # tensor sizes
        self.input_size = input_size
        self.input_condition_size = input_condition_size
        self.output_size = output_size
        # ----- #
        # unet
        self.network = get_network(input_size, input_condition_size, output_size, embedding_dims, widths, block_depth)
        # ----- #
        # EMA weighting 
        self.ema = ema
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)
        # mae loss of predicted noise 
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        # mae loss of reconstructed forward diffusion
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def diffusion_schedule(self, diffusion_times):
        # Linear schedule in angular form
        
        # diffusion times -> angles
        start_angle = tf.cast(tf.math.acos(self.max_signal_rate), "float32")
        end_angle = tf.cast(tf.math.acos(self.min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.math.cos(diffusion_angles)
        noise_rates = tf.math.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, conditions, noise_rates, signal_rates, training):
        # single reverse diffusion step
        if training:
            # the original weights for training
            network = self.network
        else:
            # EMA weights for validation
            network = self.ema_network

        # predict noise
        pred_noises = network([noisy_images, conditions, noise_rates**2], training=training)
        
        # subtrack noise from the input image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, conditions):
        # get batch size
        num_images = initial_noise.shape[0]
        step_size = 1.0 / self.diffusion_steps

        # sampling from tf.random.normal
        # in func generate
        next_noisy_images = initial_noise
        
        for step in range(self.diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1, 1)) - step * step_size
            
            # pull the current diffusion schedule
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            
            # single reverse diffusion step
            pred_noises, pred_images = self.denoise(noisy_images, conditions, 
                                                    noise_rates, signal_rates, training=False)
            
            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)

            # produce reverse diffusion output for the current step
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
            
        return pred_images

    def generate(self, num_images, conditions):
        # sample initial noise
        initial_noise = tf.random.normal(shape=(num_images,)+self.input_size)
        generated_images = self.reverse_diffusion(initial_noise, conditions)
        
        return generated_images

    def train_step(self, images_and_conditions):
        # images_and_conditions = ([image, condition])
        images = images_and_conditions[0][0]
        conditions = images_and_conditions[0][1]

        # ----- #
        # forward diffuse
        # sample noise of the batch shape
        noises = tf.random.normal(shape=tf.shape(images))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(images)[0], 1, 1, 1, 1), minval=0.0, maxval=1.0)
        
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        
        # mix the images with noises
        noisy_images = signal_rates * images + noise_rates * noises

        
        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, conditions, noise_rates, signal_rates, training=True)
            
            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric
            
        # backward step
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # update EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
            
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images_and_conditions):
        # images_and_conditions = ([image, condition])
        images = images_and_conditions[0][0]
        conditions = images_and_conditions[0][1]

        # forward diffuse
        noises = tf.random.normal(shape=tf.shape(images))
        
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(images)[0], 1, 1, 1, 1), minval=0.0, maxval=1.0)
        
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        
        # mix the images with noises
        noisy_images = signal_rates * images + noise_rates * noises

        # reverse diffusion based on the sample diffusion time step
        pred_noises, pred_images = self.denoise(
            noisy_images, conditions, noise_rates, signal_rates, training=False)

        # validation loss
        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)
        
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        
        return {m.name: m.result() for m in self.metrics}
