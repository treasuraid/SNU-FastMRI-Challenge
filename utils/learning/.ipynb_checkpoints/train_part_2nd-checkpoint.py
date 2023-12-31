import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import os 
from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet
from utils.model.kbnet import KBNet_l,  KBNet_s
from utils.model.NAFNet_arch import NAFNet
from utils.model.rcan import RCAN

from utils.data.load_data import SliceData2nd, MultiSliceData2nd
from utils.data.transforms import DataTransform2nd, MultiDataTransform2nd
import pandas as pd 
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from kornia.morphology import dilation, erosion 


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, device):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    k = torch.ones(3,3).float().to(device)


    for iter, data in enumerate(tqdm(data_loader)):
        input, target, maximum, _, _, brightness = data
        input = input.cuda(non_blocking=True)
        input.requires_grad = True
        target = target.cuda(non_blocking=True)
#         maximum = maximum.cuda(non_blocking=True)
        maximum = target.view(target.shape[0], -1).max(dim = 1).values
#         print(maximum)
        brightness = brightness.cuda(non_blocking=True)
        output = model(input)
        

#         output = output.squeeze(1)
        target = target.unsqueeze(1)
#         print(target.shape, input.shape, output.shape)
        if args.loss_mask : 
#                 print(brightness[:,None,None,None].shape)
                loss_mask  = (target > (5e-5 * brightness[:,None,None,None])).float()
                # for 1 time 
                loss_mask  = erosion(loss_mask, k)
                for i in range(15):
                    loss_mask = dilation(loss_mask, k)
                for i in range(14):
                    loss_mask = erosion(loss_mask, k)
                loss_mask = loss_mask.squeeze(0)
                output= output * loss_mask
                target = target * loss_mask 
          
        loss = loss_type(output, target, maximum) / args.grad_accumulation 
        loss.backward()  
        
#         plt.imsave(os.path.join("./garage1", str(iter) +"_" +"output.png"), output[0].cpu().detach().numpy())
#         plt.imsave(os.path.join("./garage1", str(iter) +"_" +"recon.png"), input[0,1,:,:].cpu().detach().numpy())

#         plt.imsave(os.path.join("./garage1", str(iter)) +"_" +"grappa.png", input[0,2,:,:].cpu().detach().numpy())

#         plt.imsave(os.path.join("./garage1", str(iter)) +"_" +"diff.png", np.abs(target[0].cpu().numpy() -input[0,1,:,:].cpu().detach().numpy()))
        
#         plt.imsave(os.path.join("./garage1", str(iter)) +"_" +"diff_grappa.png", np.abs(target[0].cpu().numpy() -input[0,2,:,:].cpu().detach().numpy()))
        
        
        total_loss += loss.item() * args.grad_accumulation 

        wandb.log({"batch_loss": loss.item() * args.grad_accumulation})
        
        if (((iter + 1) % args.grad_accumulation) == 0):
            if args.grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            
            optimizer.zero_grad()
            
        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, device):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()
    k = torch.ones(3,3).float().to(device)

    df = pd.DataFrame(columns = [i for i in range(0, 30)])
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input, target, _, fnames, slices, brightness = data
            
            input = input.cuda(non_blocking=True)
            output = model(input)
            brightness = brightness.cuda()
            target = target.to(device)
            
            if args.loss_mask : 
                loss_mask  = (target > 5e-5 * brightness[:,None,None,None]).float()
            # for 1 time 
                loss_mask  = erosion(loss_mask, k)
                for i in range(15):
                    loss_mask = dilation(loss_mask, k)
                for i in range(14):
                    loss_mask = erosion(loss_mask, k)
                loss_mask = loss_mask.squeeze(0)
                output= output * loss_mask
                target = target * loss_mask 
            
            target = target.squeeze(0)
            output = output.squeeze(0)
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()
            
            if (((iter+1) % args.report_interval)== 0) : 
                print(f"{iter} validated")

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
    metric_loss = 0
    num_total_slice = 0 
    for fname in reconstructions.keys():
        metric_losses = ssim_loss(targets[fname], reconstructions[fname])
        metric_loss += np.sum(metric_losses)
        num_total_slice += len(metric_losses) 
        if len(metric_losses) < 30:
            metric_losses = np.append(metric_losses, np.zeros(30 - len(metric_losses)))
        
        df.loc[fname]  = list(metric_losses) 
    
    metric_loss = metric_loss / num_total_slice
    df.to_csv(os.path.join(args.val_loss_dir, "ssim_score_{}.csv".format(args.loss_mask)))
    
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    # todo : add config file for better readability
    
    if args.model == "kbnet":
        model = KBNet_s(img_channel=args.input_channel, out_channel=1 if not args.multi_channel else 3, width=32, middle_blk_num=6, enc_blk_nums=[2, 2, 2, 2],
                    dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=True, ffn_scale=1.5).to(device=device)
    
    elif args.model == "nafnet":
        
        width = 32
        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]
        model = NAFNet(img_channel=args.input_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    elif args.model == "rcan":
        args.n_feats = 160
        args.n_resblocks = 25
        args.n_resgroups = 4 
        args.reduction = 16
        args.res_scale = 0.125
        model = RCAN(args)
    
    print(model)
    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 40, gamma= 0.2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
    
    best_val_loss = 1.
    start_epoch = 0
    

    
    if not args.multi_channel :
        train_transform = DataTransform2nd(isforward= False, max_key= args.max_key, edge = args.edge, aug = args.aug)
        val_transform = DataTransform2nd(isforward= False, max_key= args.max_key, edge = args.edge, aug = False)
        train_loader = torch.utils.data.DataLoader(SliceData2nd(args.data_path_train, 
                                args.recon_path / "reconstructions_val",
                                transform=train_transform,
                                input_key = args.input_key,
                                target_key= args.target_key),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
    
        val_loader = torch.utils.data.DataLoader(SliceData2nd(args.data_path_val,  
                                args.recon_path / "reconstructions_train",
                                transform=val_transform,
                                input_key = args.input_key,
                                target_key= args.target_key,
                                part = True),
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,)
    
    else :
        train_transform = MultiDataTransform2nd(isforward= False, max_key= args.max_key, edge = args.edge, aug = args.aug)
        val_transform = MultiDataTransform2nd(isforward= False, max_key= args.max_key, edge = args.edge, aug = False)

        train_loader = torch.utils.data.DataLoader(MultiSliceData2nd(args.data_path_train, 
                                args.recon_path / "reconstructions_train",
                                transform=train_transform,
                                input_key = args.input_key,
                                target_key= args.target_key,num_slices = 3),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
    
        val_loader = torch.utils.data.DataLoader(MultiSliceData2nd(args.data_path_val,  
                                args.recon_path / "reconstructions_val",
                                transform=val_transform,
                                input_key = args.input_key,
                                target_key= args.target_key,num_slices = 3),
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,)
    
    
    val_loss_log = np.empty((0, 2))
#     val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, device)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, device)
        
        scheduler.step()
        
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, device)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = args.val_loss_dir / "val_loss_log"
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
        

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )