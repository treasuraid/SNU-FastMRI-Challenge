"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from skimage.metrics import structural_similarity
import h5py
import numpy as np
import torch
import random

def save_reconstructions(reconstructions, out_dir, targets=None, inputs=None):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])

def ssim_loss(gt, pred, maxval=None, win_size=7, k1=0.01, k2=0.03):
    """Compute Structural Similarity Index Metric (SSIM)
       ssim_loss is defined as (1 - ssim)
    """
    maxval = gt.max() if maxval is None else maxval

    ssim = [0 for _ in range(gt.shape[0])]
    for slice_num in range(gt.shape[0]):
        ssim[slice_num] = structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval, win_size=win_size, K1=k1, K2=k2
        )
        
    return 1 - np.array(ssim)

def seed_fix(n):
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(n)
    random.seed(n)


def print_model_num_parameters(model : torch.nn.Module):
    """
    Print the number of parameters in a model
    """
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_parameters}")

    return num_parameters


def print_model_parameters(model : torch.nn.Module, only_trainable : bool = True):
    """
    Print the parameters in a model
    """
    for name, param in model.named_parameters():
        if param.requires_grad or not only_trainable:
            print(name, param.data)


