# Improving ensemble extreme precipitation forecasts using generative artificial intelligence

Yingkai Sha, Ryan A. Sobash, David John Gagne II

NSF National Center for Atmospheric Research, Boulder, Colorado, USA

Sha, Y., Sobash, R.A. and Gagne II, D.J., 2024. Improving ensemble extreme precipitation forecasts using generative artificial intelligence. arXiv preprint arXiv:2407.04882. url: https://arxiv.org/abs/2407.04882

## Abstract

An ensemble post-processing method is developed to improve the probabilistic forecasts of extreme precipitation events across the conterminous United States (CONUS). The method combines a 3-D Vision Transformer (ViT) for bias correction with a Latent Diffusion Model (LDM), a generative Artificial Intelligence (AI) method, to post-process 6-hourly precipitation ensemble forecasts and produce an enlarged generative ensemble that contains spatiotemporally consistent precipitation trajectories. These trajectories are expected to improve the characterization of extreme precipitation events and offer skillful multi-day accumulated and 6-hourly precipitation guidance. The method is tested using the Global Ensemble Forecast System (GEFS) precipitation forecasts out to day 6 and is verified against the Climate-Calibrated Precipitation Analysis (CCPA) data. Verification results indicate that the method generated skillful ensemble members with improved Continuous Ranked Probabilistic Skill Scores (CRPSSs) and Brier Skill Scores (BSSs) over the raw operational GEFS and a multivariate statistical post-processing baseline. It showed skillful and reliable probabilities for events at extreme precipitation thresholds. Explainability studies were further conducted, which revealed the decision-making process of the method and confirmed its effectiveness on ensemble member generation. This work introduces a novel, generative-AI-based approach to address the limitation of small numerical ensembles and the need for larger ensembles to identify extreme precipitation events.

## Data
* The forecast to post-process:
  * GEFS version 12 6 hourly accumulated total precipitation (APCP)
* Training and verification target:
  * Climatology-Calibrated Precipitation Analysis (CCPA)

## Method
* Vector Quantisation Variational Autoencoder (VQ-VAE)
* 3-D Vision Transformer (ViT)
* Latent Diffusion Model (LDM)

## Navigation
* Model builder: [model_utiles.py](https://github.com/yingkaisha/AIES_24_0063/blob/main/libs/model_utils.py)
* Model summary: [MODEL03_all_model_summary.ipynb](https://github.com/yingkaisha/AIES_24_0063/blob/main/MODEL03_all_model_summary.ipynb)
* Model inference pipeline: [MODEL03_model_full_pipeline.ipynb](https://github.com/yingkaisha/AIES_24_0063/blob/main/MODEL03_model_full_pipeline.ipynb)
* Model training: [VQ-VAE](https://github.com/yingkaisha/AIES_24_0063/blob/main/MODEL00_VQ_VAE_main.ipynb), [VQ-VAE output section](https://github.com/yingkaisha/AIES_24_0063/blob/main/MODEL00_VQ_VAE_refine.ipynb), [3-D ViT](https://github.com/yingkaisha/AIES_24_0063/blob/main/MODEL01_ViT_48h_pretrain.ipynb), [LDM](https://github.com/yingkaisha/AIES_24_0063/blob/main/MODEL02_LDM_3d_main.ipynb)
* Explainability study: [RESULT04_Latent_space_verif.ipynb](https://github.com/yingkaisha/AIES_24_0063/blob/main/RESULT04_Latent_space_verif.ipynb), [PLOT_Latent_space_vis.ipynb](https://github.com/yingkaisha/AIES_24_0063/blob/main/PLOT_Latent_space_vis.ipynb)
* Data visualization and results: [Example case study](https://github.com/yingkaisha/AIES_24_0063/blob/main/PLOT_example.ipynb), [Reliability diagrams on 40 mm and 99th percentile thresholds](https://github.com/yingkaisha/AIES_24_0063/blob/main/PLOT_reliability_diagrams.ipynb), [Reliability diagrams on more fixed thresholds](https://github.com/yingkaisha/AIES_24_0063/blob/main/PLOT_reliability_diagrams_more_thres.ipynb), [Brier scores with varying ensemble sizes](https://github.com/yingkaisha/AIES_24_0063/blob/main/PLOT_BSS_ens_members.ipynb)
* Comparisons between CCPA and ERA5 total precipitation based on reviewer feedback [Comparison results](https://github.com/yingkaisha/AIES_24_0063/blob/main/REVIEW_PLOT_data_analysis.ipynb)
