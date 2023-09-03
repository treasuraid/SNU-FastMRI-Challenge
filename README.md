# SNU FastMRI challenge 2023 - 2nd place solution for the FastMRI challenge 2023

# Team members 
- [Jinpil Choi] (Seoul National University)

# Introduction
My Solution consist of 3 training steps.

1. Pre-training E2E-Varnet(6 cacacades, 13 unet channels, 4 sensitivity map channels) with total 5674 slices, 357 training patients mri data.

2. Fine-tuning E2E-Varnet with extra 2 cascades on 8x undersampled k-space data.

3. Training Image-to-Image reconstruction model (varnet, kbnet) that gets reconstruction iamge from trained E2E-Varnet and Grappa reconstruction image and original image.

Improves the performance as training step goes on.

# Requirements 
```
pip install -r requirements.txt 
```

# How to run codes? 

## Training 
```
train.py --batch-size 1 --lr 0.001 --seed 76 -r 100 -n varnet_6_13_4_lm --chans 13 --cascade 6 --sens_chans 4 --grad_norm 1 --grad_accumulation 1 --aug_delay 4 --loss_mask --scheduler step --aug_strength 0.5 --aug_max_rotation 180 --aug_max_shearing_x 15.0 --aug_max_shearing_y 15.0 --step_size 10 --aug
```

## Reconstruction 
```
reconstruct.py -n varnet_6_13_4_lm -m acc4 
reconstruct.py -n varnet_6_13_4_lm -m acc8
```

## Evaluation
```
train_2nd.py -n nafnet_vanilla_from_8 --seed 2023 --report-interval 100 --lr 0.0003 --batch-size 2 --grad_accumulation 16 --model nafnet --recon-path ../result/varnet_8_13_4_resume_base_8x/ --loss_mask True --grad_norm 0.01
```