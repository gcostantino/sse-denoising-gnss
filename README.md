[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://opensource.org/licenses/GPL-3.0)

# sse-denoising-gnss

Source code for the paper "Denoising of Geodetic Time Series Using Spatiotemporal Graph Neural Networks: Application to Slow Slip Event Extraction".
Please cite the original paper when using this code:

Costantino, G., Giffard-Roisin, S., Mura, M. D., & Socquet, A. (2024). Denoising of Geodetic Time Series Using Spatiotemporal Graph Neural Networks: Application to Slow Slip Event Extraction. arXiv preprint arXiv:2405.03320.

## Installation

Use the file ```requirements.txt``` to install the correct dependencies for the project.

## How the code works

The code follows a sequence of steps:

1. Generation of the synthetic database: ```generate_synthetic_data.py```
2. Training of the model: ```train.py```
3. Inference
    - inference on the synthetic database: ```inference_synthetic_data.py```
    - inference on real GNSS data: ```inference_running_window.py```
4. Test: tables and figures
    - test on the synthetic database: ```test_synthetic_data.py```
    - test on real GNSS data: ```test_real_data.py```

Additional files:
- ```ablation_study.py```: reproduces the figures and results of the ablation study (Table II and Fig. 4 of the paper)
- ```adj_matrix_plots.py```: reproduces the figures on the optimal graph connections (Fig. 5 of the paper)

**Note well**: in order to reproduce the results presented in the paper (i.e., steps 1 to 3 should be skipped), the synthetic database as well as the prediction files can be downloaded at: [10.5281/zenodo.11283069](https://doi.org/10.5281/zenodo.11283069)

After downloading, the file placement should look like the following:

```
denois_synth_ts_cascadia_realgaps_extended_v5_200stations_6_7_depth_20_40.data
predictions
   ablation
      pred_denoising_test_data_ablation_notransf
      pred_denoising_test_data_ablation_spatial_att_only
      pred_denoising_test_data_ablation_temp_att_only
   pred_denoising_test_data
   pred_denoising_test_data_1d_xue_freymueller
   pred_denoising_test_data_2d_unet

```
