import parser
import os
from load_data import create_data_loaders
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    args = parser.ArgumentParser(description='PyTorch Variational Network MRI Reconstruction')
    args.add_argument('--batch_size', type=int, default=1, help='Batch size should be 1')
    args.add_argument('--input_key', type=str, default='kspace', help='Name of input key')
    args.add_argument('--target_key', type=str, default='image_label', help='Name of target key')
    args.add_argument('--max_key', type=str, default='max', help='Name of max key')
    args.add_argument('--data_path', type=str, default='root/Data/leaderboard/', help='Directory of test data')
    parser.add_argument('-m', '--mask', type=str, default='acc4', choices=['acc4', 'acc8'], help='type of mask | acc4 or acc8')



    args = args.parse_args()

    forward_loader = create_data_loaders(data_path=args.data_path, args = args, isforward = True)

    for i, data in enumerate(forward_loader):
        for (mask, kspace, image, _, _, fnames, slices) in forward_loader:
            # visualize mask and image in 1 frame
            # kspace
            mask = mask.numpy()
            # expand mask to 2d array for visualization
            mask = mask.flatten()
            mask = np.vstack([mask] * kspace.shape[-3])

            print(np.sum(mask))

            # visualize mask

