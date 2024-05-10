# Precipitation forecast post-processing and ensemble member generation using latent diffusion model

This project aims to bias correct and calibrate Global Forecast System (GFS) precipitation forecasts in the Conterminous United States (CONUS) by using Latent Diffusion Model (LDM). The post-processing is probabilistic, it generates ensemble members from the given GFS determinstic forecasts.

## Data
* The forecast to post-process:
  * GFS 6 hourly accumulated total precipitation (APCP) up to 144 hours
* Learning target:
  * Climatology-Calibrated Precipitation Analysis (CCPA) 6 hourly quantitative precipitation estimation

## Method
The project containts three neural networks: 

* Vector Quantisation Variational Autoencoder (VQ-VAE)
* 3-D Vision Transformer (ViT)
* LDM

## Note
The project is in its early stage.

