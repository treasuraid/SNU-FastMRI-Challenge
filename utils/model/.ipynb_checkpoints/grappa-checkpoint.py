from pygrappa import grappa

import numpy as np
import pygrappa

from fastmri import ifft2c
import fastmri
import matplotlib.pyplot as plt 
import torch 


def grappa_recon(kspace: np.ndarray, mask: np.ndarray):
    """
    Args:
        kspace: (C, H, W)
        mask: (W)
    """
    h, w = kspace.shape[-2:]
    squeezed_mask = mask.squeeze()
    cent = squeezed_mask.shape[0] // 2
    # running argmin returns the first non-zero
    left = np.argmin(np.flip(squeezed_mask[:cent], axis = 0), axis=0)
    right = np.argmin(squeezed_mask[cent:], axis=0)
    num_low_freqs = min(left, right)
    calib = kspace[:, :, cent - num_low_freqs:cent + num_low_freqs].copy()
    grappa_result = grappa(kspace, calib, kernel_size=(7, 7), coil_axis=0)
    
    grappa_result = torch.from_numpy(np.stack((grappa_result.real, grappa_result.imag), axis = -1))
    grappa_result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(grappa_result)), dim=0)     
    return grappa_result[..., h//2-192:h//2+192, w//2-192:w//2+192].numpy()
    

if __name__ == '__main__':

    import h5py
    import argparse
    import os 
    from tqdm import tqdm
    import matplotlib.pyplot as plt 
    # 3 times 반복
    data_dir = "../../../../Data/val/kspace" 
    grappa_dir = "../../../../Data/grappa/val/" 
    original_dir = "../../../../Data/val/image"
    
    for fname in tqdm(os.listdir(data_dir)) :
        
        # open h5 file
        
        with h5py.File(os.path.join(data_dir, fname), 'r') as f :
            image_masked = np.array(f["kspace"])
            num_slice = image_masked.shape[0]
            mask = np.array(f["mask"])
            # stack mask with kspace.shape[-2] 
            
            
            grappa_recon_images = []
            for slice in tqdm(range(num_slice)) :
                kspace_slice = image_masked[slice]
                grappa_recon_image = grappa_recon(kspace_slice*mask, mask)
                original_image = grappa_recon(kspace_slice, mask)

#                 plt.imsave(fname + "_" + str(slice) + "_7.png",grappa_recon_image)
#                 plt.imsave(fname + "_" + str(slice) + "_origin.png",original_image)

                grappa_recon_images.append(grappa_recon_image)
                
            grappa_recon_images = np.array(grappa_recon_images)
            
            # save grappa_recon_images
            
            with h5py.File(os.path.join(grappa_dir, fname), 'w') as f :
                f.create_dataset("image_grappa", data = grappa_recon_images)
                
            
            # test open 
            
            with h5py.File(os.path.join(grappa_dir, fname), 'r') as f :
                
                print(np.array(f["image_grappa"]).shape)