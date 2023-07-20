from pygrappa import grappa

import numpy as np
import pygrappa

def iff2c(kspace) :
    h, w = kspace.shape[-2:]
    return np.roll(np.roll(np.fft.ifft2(kspace), shift=h // 2, axis=1),
                   shift=w // 2, axis=2)

def grappa_recon(kspace: np.ndarray, mask: np.ndarray):
    """
    Args:
        kspace: (C, H, W)
        mask: (1, 1, H, W, 1)
    """
    h, w = kspace.shape[-2:]
    squeezed_mask = mask.squeeze()
    cent = squeezed_mask.shape[0] // 2
    # running argmin returns the first non-zero
    left = np.argmin(np.flip(squeezed_mask[:cent], axis = 0), axis=0)
    right = np.argmin(squeezed_mask[cent:], axis=0)
    num_low_freqs = min(left, right)
    calib = kspace[:, :, cent - num_low_freqs:cent + num_low_freqs].copy()
    grappa_result = grappa(kspace, calib, kernel_size=(5, 5), coil_axis=0)
    return iff2c(grappa_result)


if __name__ == '__main__':

    import h5py
    import matplotlib.pyplot as plt
    debug_datas = ["/Users/a./fast_mri/data_debug/brain_acc4_101.h5", "/Users/a./fast_mri/data_debug/brain_acc8_101.h5"]

    for fname in debug_datas :
        with h5py.File(fname, "r") as hf:

            kspace = np.array(hf["kspace"])[0]
            mask = np.array(hf["mask"])
            grappa_reconstruction = grappa_recon(kspace * mask, mask)
            print(grappa_reconstruction.shape)
            plt.imshow(np.sqrt(np.sum(np.abs(kspace * mask != 0)**2, axis=0)))
            plt.show()

            plt.subplot(1, 4, 1)
            plt.imshow(np.sqrt(np.sum(np.abs(grappa_reconstruction)**2, axis=0)))
            plt.subplot(1, 4, 2)
            plt.imshow(np.sqrt(np.sum(np.abs(iff2c(kspace * mask))**2, axis=0)))

            plt.subplot(1, 4, 3)
            plt.imshow(np.sqrt(np.sum(np.abs(iff2c(kspace))**2, axis=0)))
            plt.subplot(1, 4, 4)
            # plot diff
            plt.imshow(np.sqrt(np.sum(np.abs(iff2c(kspace))**2, axis=0) - np.sum(np.abs(grappa_reconstruction)**2, axis=0)))
            plt.show()
    #
    # binary_mask = np.zeros((1, 1, 256, 256, 1))
    # binary_mask[..., 128-30:128+40, :] = 1
    #
    # grappa_reconstructed = grappa_recon(kspace, binary_mask)
    # print(grappa_reconstructed.shape)