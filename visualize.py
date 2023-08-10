import os

import numpy as np
from matplotlib import pyplot as plt
from kornia.morphology import dilation, erosion 
import torch 
from train import parse

from utils.data.load_data import create_data_loaders
from pathlib import Path

import h5py
if __name__ == '__main__':


    # recon_dir = "../result/varnet_16_fix_lossmask/reconstructions_val"
    # val_dir = "../../Data/val/image"
    
    # k = torch.ones(3,3).float()
    # for fname in os.listdir(val_dir) :
        
    #     with h5py.File(os.path.join(val_dir, fname)) as hf :
    #         print(hf.keys())
            
    #         for i, slice in enumerate(hf["image_label"]) :
                
    #             # plt.imsave(os.path.join("./garage2", fname[:-3] + ".png"), slice)
                
    #             # erosion and dilation
    #             loss_mask = torch.from_numpy(slice > 0.00001).float().unsqueeze(0).unsqueeze(0)
    #             loss_mask  = erosion(loss_mask, k)
    #             for i in range(10):
    #                 loss_mask = dilation(loss_mask, k)
    #             for i in range(9):
    #                 loss_mask = erosion(loss_mask, k)
    #             loss_mask = loss_mask.squeeze(0).squeeze(0).numpy()
                
    #             plt.imsave(os.path.join("./garage2", fname[:-3] +  "_" + str(i) + "_mask_ero1.png"), (loss_mask * slice) > 0)
    
    args = parse() 
    
    import utils.model.fastmri as fastmri
    
    train_dataset, train_loader = create_data_loaders(data_path=Path("/Data/val"), args=args, shuffle=True, aug=True)
    from tqdm import tqdm
    k = torch.ones(3,3).float()
    for iter, data in enumerate(tqdm(train_loader)):
        mask, kspace, kspace_origin, target, edge, maximum, fname, slice, = data
        

        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1) 
        
        # crop to 384 x 384 
        result = result[..., result.shape[-2] //2 - 192 : result.shape[-2] //2 + 192, result.shape[-1] //2 - 192 : result.shape[-1] //2 + 192]
        
        loss_mask  = (target > 5e-5).float().unsqueeze(0)
            # for 1 time 
        loss_mask  = erosion(loss_mask, k)
        # for 15 times dilation 
        for i in range(15):
            loss_mask = dilation(loss_mask, k)
        for i in range(14):
            loss_mask = erosion(loss_mask, k)
            
            
        

        plt.imsave(os.path.join("./garage1", fname[0][:-3] + "_" + str(slice.cpu().item()) +  "_5mask.png"), loss_mask.squeeze(0).squeeze(0).numpy())
        
        loss_mask  = (target > 2e-5).float().unsqueeze(0)
            # for 1 time 
        loss_mask  = erosion(loss_mask, k)
        # for 15 times dilation 
        for i in range(15):
            loss_mask = dilation(loss_mask, k)
        for i in range(14):
            loss_mask = erosion(loss_mask, k)
            
        plt.imsave(os.path.join("./garage1", fname[0][:-3] + "_" + str(slice.cpu().item()) +  "_2mask.png"), loss_mask.squeeze(0).squeeze(0).numpy())
        
        target = target.numpy()  
        plt.imsave(os.path.join("./garage1", fname[0][:-3] + "_" + str(slice.cpu().item()) +  ".png"), target[0])
        
        
        plt.imsave(os.path.join("./garage1", fname[0][:-3] + "_" + str(slice.cpu().item()) + "_recon.png"), result[0])