import argparse
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import *
from utils.common.loss_function import SSIMLoss, EdgeMAELoss, FocalFrequencyLoss, MS_SSIM 
from utils.model.varnet import VarNet, VarnetAdded
from utils.model.EAMRI import EAMRI
# from utils.model.test_direct import CustomMutiDomainNet 
import os
import torch.nn.functional as F

from kornia.morphology import dilation, erosion 

from logging import getLogger

import wandb ## for logging
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type, device=torch.device('cuda:0')):
    logger.warning(f"Running Training Epoch {epoch}")

    model.train()
    start_epoch = start_iter = time.perf_counter()
    ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0) 
    len_loader = len(data_loader)
    total_loss = 0.
    
    # for masking loss 
    k = torch.ones(3,3).float().to(device)
    
    for iter, data in enumerate(tqdm(data_loader)):
        mask, kspace, kspace_origin, target, edge, maximum, fname, _, = data
#         print(mask.shape, kspace.shape, target.shape, maximum, torch.sum(mask > 0))
        mask = mask.to(device)
        kspace = kspace.to(device)
        target = target.to(device)
        # use gt.max 
#         maximum = maximum.to(device) 
        maximum = target.max().view(-1,1,1,1)
        # kspace_origin = kspace_origin.to(device)
        kspace.requires_grad = True
        # mask.requires_grad = True
        # target.requires_grad = True
        
        
        output_image = model(kspace, mask)

        if args.model == "eamri": 
            output_edges, output_image = output_image[0], output_image[1]
        if args.loss_mask:
            loss_mask  = (target > 5e-5).float().unsqueeze(0)
            # for 1 time 
            loss_mask  = erosion(loss_mask, k)
            # for 15 times dilation 
            for i in range(15):
                loss_mask = dilation(loss_mask, k)
            for i in range(14):
                loss_mask = erosion(loss_mask, k)
            
            # erosion by 1 time, dilate for 15 times and erosion by 1 
            loss_mask = loss_mask.float().squeeze(0)
            output_image = output_image * loss_mask
            target = target * loss_mask
            
            if args.model == "eamri":
                output_edges = output_edges * loss_mask
                edge = edge * loss_mask 

        if args.model == "eamri":
            print("output_image", output_image.shape)
            loss = loss_type((output_edges, output_image), (target, edge), maximum)
            loss_ssim = SSIMLoss()(output_image, target, maximum)
            wandb.log({"train_ssim_loss" : loss_ssim.item()})
        else :
            loss  = loss_type(output_image, target, maximum) 
        # loss_fft = torch.nn.functional.l1_loss(output_image, target) * 5000
        # if loss.item() > 0.04 :
        #     print(f"loss {loss.item()} in {fname}")
        loss = loss / args.grad_accumulation # Normalize our loss (if averaged) by grad_accumulation
        loss.backward()
        
        if ((iter + 1) % args.grad_accumulation) == 0:
            if args.grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            
            optimizer.zero_grad()

        total_loss += loss.item() * args.grad_accumulation
        wandb.log({"train_batch_loss" : loss.item() * args.grad_accumulation,
                   "learning_rate" : optimizer.param_groups[0]['lr']})

        if (iter % args.report_interval) == 0:
            logger.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, device=torch.device("cuda:0")):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()
    k = torch.ones(3,3).float().to(device)
    # maintain the csv file for ssim score
    df = pd.DataFrame(columns = [i for i in range(0, 30)])
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            mask, kspace, _, target, edge, _, fnames, slices = data
            mask = mask.to(device)
            kspace = kspace.to(device)
            target = target.to(device)

            if args.model == "eamri":
                edge_output, output = model(kspace, mask) 
            else :
                output = model(kspace, mask)
            if True:
                loss_mask  = (target > 5e-5).float().unsqueeze(0)
            # for 1 time 
                loss_mask  = erosion(loss_mask, k)
                for i in range(15):
                    loss_mask = dilation(loss_mask, k)
                for i in range(14):
                    loss_mask = erosion(loss_mask, k)
                loss_mask = loss_mask.squeeze(0)
                output= output * loss_mask
                target = target * loss_mask 
            
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

            if (iter % args.report_interval == 0 and (iter > 0)):
                print(f"{iter} validation done")
                

    for fname in reconstructions:
        reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname].items())])
    for fname in targets:
        targets[fname] = np.stack([out for _, out in sorted(targets[fname].items())])
    
    # person mean -> slice mean -> ssim 
    # key -> list of ssim scoreum
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
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    model_name = f"model{epoch}.pt"

    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / model_name
    )
    if is_new_best:
        shutil.copyfile(exp_dir / model_name, exp_dir / 'best_model.pt')


def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)
    # logger.info("Current cuda device: %d", torch.cuda.current_device())

    # Choose Model
    if args.model == 'varnet':
        logger.warning("model: varnet")
        model = VarNet(num_cascades=args.cascade,
                       chans=args.chans,
                       sens_chans=args.sens_chans,
                       unet=args.unet,
                       config=args.config)

    elif args.model == 'eamri':
        logger.warning("model: eamri")
        model = EAMRI(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1, 3, 3, 3, 3],
                      fNums=[48, 96, 96, 96, 96], n_MSRB=3, shift=True)

    elif args.model == "varnet_add":
        logger.warning("model: varnet_add")
        model = VarnetAdded(num_cascades=args.cascade)
        
    elif args.model == "mdnet":
#         model = CustomMutiDomainNet()
        exit(1)
    else:
        raise NotImplementedError("model not found")

    print_model_num_parameters(model)
    
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # get scheduler
    if args.scheduler is not None:
        if args.scheduler == "cosine":
            print("use cosineanealingLR")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        elif args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        else:
            raise NotImplementedError("scheduler not found")
    else:
        scheduler = None

    if args.resume_from is not None :
        resume_from(model, optimizer, args.resume_from, device=device)


    # get loss function
    if args.loss == "ssim":
        loss_type = SSIMLoss().to(device=device)
    elif args.loss == "mse":
        loss_type = torch.nn.MSELoss().to(device=device)
    elif args.loss == "edge":
        loss_type = EdgeMAELoss().to(device=device)
    elif args.loss == "ms-ssim":
        loss_type = MS_SSIM
    else:
        raise NotImplementedError("loss not found")

    best_val_loss = 1.
    start_epoch = 0
    val_loss_log = np.empty((0, 2))
    
    # mix val data file path and train data file path for k-fold validation
    
    data_files = []
    data_files += [Path(os.path.join(args.data_path_train / "kspace", file)) for file in sorted(os.listdir(args.data_path_train / "kspace")) if "acc4" in file] 
    
    print(data_files)
    
    
    np.random.seed(args.data_seed) 
    np.random.shuffle(data_files) 
    
    # shuffle data_files 80% train, 20% val
    
    # seed fixing

    # split data_files by args.data_split_num [0,1,2,3,4] # to avoid data leakage
    
    # 0 -> val ~ 0.2 , train ~ 0.8 
    # 1 -> val ~ 0.2 ~ 0.4  , train except it 
    # 2 -> val ~ 0.4 ~ 0.6 , train except it
    # 3 -> val ~ 0.6 ~ 0.8 , train except it
    # 4 -> val ~ 0.8 ~ 1.0 , train except it
    
    val_data_path = data_files[int(len(data_files) * args.data_split_num / 3) : int(len(data_files) * (args.data_split_num + 1) / 3)]
    train_data_path = data_files[:int(len(data_files) * args.data_split_num / 3)] + data_files[int(len(data_files) * (args.data_split_num + 1) / 3):]
    
    val_data_path += [Path(str(file).replace("acc4", "acc8")) for file in val_data_path]
    train_data_path += [Path(str(file).replace("acc4", "acc8")) for file in train_data_path]
    
    train_data_path += [Path(os.path.join(args.data_path_val / "kspace", file)) for file in sorted(os.listdir(args.data_path_val / "kspace"))] 
    
    if args.data_split_num == 1 :
        train_data_path += [Path(os.path.join(args.data_path_train / "kspace", "brain_acc8_179.h5"))]
    else :
        val_data_path += [Path(os.path.join(args.data_path_train / "kspace", "brain_acc8_179.h5"))]
    
    train_dataset, train_loader = create_data_loaders(data_path=train_data_path, args=args, shuffle=True, aug=args.aug)
    _ , val_loader = create_data_loaders(data_path=val_data_path, args=args, shuffle=False, aug= False)
    
    
#     # test saving and validation
    # save_model(args, args.exp_dir, 0, model, optimizer, best_val_loss, False)
    val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
    # print(val_loss)
    for epoch in range(start_epoch, args.num_epochs):

        logger.warning(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        logger.warning(f"Actual Batch size: {args.batch_size} * {args.grad_accumulation} = "
                    f"{args.batch_size * args.grad_accumulation}")

        # train
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, scheduler, loss_type, device)

        if args.scheduler is not None:
            scheduler.step()
        # validate
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, device)


        # cal loss to tensor
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        # num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)
        # val_loss = val_loss / num_subjects
        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        
        # logging at wandb
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        # save loss
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss.item()]]), axis=0)
        logger.warning(f"val loss graph : {val_loss_log}")

        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        logger.info(f"loss file saved! {file_path}")
        
        # save model
        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        logger.warning(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        # save reconstructions if new best
        if is_new_best:
            logger.warning("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            logger.warning(f'ForwardTime = {time.perf_counter() - start:.4f}s')

        if args.aug :
            train_dataset.transform.augmentor.current_epoch += 1

def load_model(args, model: torch.nn.Module):
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"

    def download_model(url, fname):
        response = requests.get(url, timeout=10, stream=True)

        chunk_size = 8 * 1024 * 1024  # 8 MB chunks
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            desc="Downloading state_dict",
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
        )

        with open(fname, "wb") as fh:
            for chunk in response.iter_content(chunk_size):
                progress_bar.update(len(chunk))
                fh.write(chunk)

    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.', 2)[1].isdigit() and (args.cascade <= int(layer.split('.', 2)[1]) <= 11):
            del pretrained[layer]
    model.load_state_dict(pretrained)

def resume_from(model, optimizer, ckpt_path, device : torch.device = torch.device('cuda:0')):

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict = False)
#     optimizer.load_state_dict(checkpoint['optimizer'])
    
    logger.warning("resume from ", ckpt_path)
    logger.warning("resume from epoch ", checkpoint['epoch'])
    logger.warning("resume from best_val_loss ", checkpoint['best_val_loss'])


    return model, optimizer
