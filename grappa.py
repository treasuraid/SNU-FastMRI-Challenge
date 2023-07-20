from pygrappa import grappa

import numpy as np
import pygrappa
def grappa_recon(kspace: np.ndarray, mask: np.ndarray):
    """
    Args:
        kspace: (1, C, H, W, 2)
        mask: (1, 1, H, W, 1)
    """
    squeezed_mask = mask[:, 0, 0, :, 0]
    cent = squeezed_mask.shape[1] // 2
    # running argmin returns the first non-zero
    left = np.argmin(np.flip(squeezed_mask[:, :cent], axis = 1), axis=1)[0]
    right = np.argmin(squeezed_mask[:, cent:], axis=1)[0]
    num_low_freqs = min(left, right)
    calib = kspace[..., cent - num_low_freqs:cent + num_low_freqs, :].copy()

    return grappa(kspace, calib, kernel_size=(5, 5), coil_axis=-4) # B C H W 2


if __name__ == '__main__':

    import h5py
    import matplotlib.pyplot as plt
    debug_datas = ["./data_debug/brain_acc4_101.h5", "./data_debug/brain_acc8_101.h5"]

    for fname in debug_datas :
        with h5py.File(fname, "r") as hf:
            kspace = np.array(hf["kspace"][0])
            mask = np.array(hf["mask"][0])
            print(kspace.shape, mask.shape)
            grappa_reconstruction = grappa_recon(kspace, mask)
            print(grappa_reconstruction.shape)

            ## plt
            plt.plot(grappa_reconstruction)
    #
    #
    # binary_mask = np.zeros((1, 1, 256, 256, 1))
    # binary_mask[..., 128-30:128+40, :] = 1
    #
    # grappa_reconstructed = grappa_recon(kspace, binary_mask)
    # print(grappa_reconstructed.shape)