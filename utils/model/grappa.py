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
    grappa_result = grappa(kspace, calib, kernel_size=(3, 3), coil_axis=0)
    grappa_result = iff2c(grappa_result)

    # crop to [384, 384]
    grappa_result = np.sqrt(np.sum(np.abs(grappa_result)**2, axis=0))
    return grappa_result[..., h//2-192:h//2+192, w//2-192:w//2+192]
    

if __name__ == '__main__':

    import h5py
    import argparse
    import os 
    from tqdm import tqdm
    import matplotlib.pyplot as plt 
    # 3 times 반복
    data_dir = "../../../../Data/train/kspace" 
    grappa_dir = "../../../../Data/grappa/val/" 
    
    
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
                print(grappa_recon_image.shape, np.max(grappa_recon_image), np.min(grappa_recon_image))
                print(original_image.shape, np.max(original_image), np.min(original_image))
                grappa_recon_images.append(grappa_recon_image)
                
            grappa_recon_images = np.array(grappa_recon_images)
            
            # save grappa_recon_images
            
            with h5py.File(os.path.join(grappa_dir, fname), 'w') as f :
                f.create_dataset("image_grappa", data = grappa_recon_images)
                
            
            # test open 
            
            with h5py.File(os.path.join(grappa_dir, fname), 'r') as f :
                
                print(np.array(f["image_grappa"]).shape)