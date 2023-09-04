# SNU FastMRI challenge 2023 - 2nd place solution  

## Team members 
- [Jinpil Choi] (Seoul National University)

## Introduction
My Solution consist of 3 training steps.

1. Pre-training E2E-Varnet(6 cacacades, 13 unet channels, 4 sensitivity map channels) with total 5674 slices, 357 training patients mri data.
1.1 Utilized various masking strategy from 4x to 8x undersampling for learning general reconstruction model.
1.2 Augmentation with random rotation, shearing, scaling, flipping, and random noise from MRAgument paper.
1.3 Utilized loss masking strategy with minimum threshold 5e-5.
1.4 Utilized gradient accumluation, clipping norm.
  
2. Fine-tuning E2E-Varnet with extra 2 cascades on 8x undersampled k-space data.
2.1 added extra layer for 8x undersampling.
2.2 To avoid overfitting or catastrophic forgetting, utilized 2nd training step with 8x undersampled data.

3. Training Image-to-Image reconstruction model (varnet, kbnet) that gets reconstruction iamge from trained E2E-Varnet and Grappa reconstruction image and original image.
3.1 Inject information of edges from grappa reconstruction image to E2E-Varnet.


Improves the performance as training step goes on.

## Requirements 
```
pip install -r requirements.txt 
```

## How to run codes? 

### Training 
```
train.py --batch-size 1 --lr 0.001 --seed 76 -r 100 -n varnet_6_13_4_lm --chans 13 --cascade 6 --sens_chans 4 --grad_norm 1 --grad_accumulation 1 --aug_delay 4 --loss_mask --scheduler step --aug_strength 0.5 --aug_max_rotation 180 --aug_max_shearing_x 15.0 --aug_max_shearing_y 15.0 --step_size 10 --aug
```

### Reconstruction 
```
reconstruct.py -n varnet_6_13_4_lm -m acc4 
reconstruct.py -n varnet_6_13_4_lm -m acc8
```

### Evaluation
```
train_2nd.py -n nafnet_vanilla_from_8 --seed 2023 --report-interval 100 --lr 0.0003 --batch-size 2 --grad_accumulation 16 --model nafnet --recon-path ../result/varnet_8_13_4_resume_base_8x/ --loss_mask True --grad_norm 0.01
```

### Ablation Study

1. Slice & SSIM loss

- Lower slice number had complex artifacts of brain and as slice number goes up, artifacts are reduced and masking for loss gets smaller.

- So SSIM value was higher as slice number goes up.

2. Edge loss of grappa reconstruction image and original image and E2E-Varnet reconstruction image 
