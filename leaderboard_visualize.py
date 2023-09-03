import argparse
import numpy as np
import h5py
import random
import glob
import os
import torch
from utils.common.loss_function import SSIMLoss
import torch.nn.functional as F
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

def forward(args):

    idx = 0
    for i_subject in range(15):
        l_fname = os.path.join(args.leaderboard_data_path, 'brain_test' + str(i_subject + 1) + '.h5')
        y_fname = os.path.join(args.your_data_path, 'brain_test' + str(i_subject + 1) + '.h5')
        with h5py.File(l_fname, "r") as hf:
            num_slices = hf['image_label'].shape[0]
        for i_slice in range(num_slices):
            with h5py.File(l_fname, "r") as hf:
                # already SSIM score is calculated with mask
                target = hf['image_label'][i_slice]
                mask = np.zeros(target.shape)
                mask[target > 3e-5] = 1
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=15)
                mask = cv2.erode(mask, kernel, iterations=14)
                maximum = hf.attrs['max']

            with h5py.File(y_fname, "r") as hf:
                recon = hf[args.output_key][i_slice]

            # visualize recon - target


#             plt.imsave(os.path.join("./garage1", 'brain_test' + str(i_subject + 1) + "_" + str(i_slice) +"_" +"target.png"), target*mask)

#             plt.imsave(os.path.join("./garage1", 'brain_test' + str(i_subject + 1) + "_" + str(i_slice) +"_" +"recon.png"), recon*mask)
 
            plt.imsave(os.path.join("./garage1", 'brain_test' + str(i_subject + 1) + "_" + str(i_slice) +"_" +"diff_non_abs.png"), np.abs((recon-target)*mask))

            idx += 1

if __name__ == '__main__':
    """
    Image Leaderboard Dataset Should Be Utilized
    For a fair comparison, Leaderboard Dataset Should Not Be Included When Training. This Is Of A Critical Issue.
    Since This Code Print SSIM To The 4th Decimal Point, You Can Use The Output Directly.
    """
    parser = argparse.ArgumentParser(description=
                                     'FastMRI challenge Leaderboard Image Evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--GPU_NUM', type=int, default=0)
    parser.add_argument('-lp', '--leaderboard_data_path', type=Path, default='/Data/leaderboard/acc4/image')
    """
    Modify Path Below To Test Your Results
    """
    parser.add_argument('-yp', '--your_data_path', type=Path,
                        default='./leaderboard_recon/')
    parser.add_argument('-m', '--mask', type=str, default='acc4', choices=['acc4', 'acc8'],
                        help='type of mask | acc4 or acc8')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')

    args = parser.parse_args()

    print(1e-5 == 0.00001)
    print(5e-5)
    forward(args)

