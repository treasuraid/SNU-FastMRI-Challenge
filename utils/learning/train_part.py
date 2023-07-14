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

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import *
from utils.common.loss_function import SSIMLoss, EdgeMAELoss
from utils.model.varnet import VarNet
from utils.model.EAMRI import EAMRI
import os

from logging import getLogger

logger = getLogger(__name__)
logger.setLevel(logging.INF0)


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type, device=torch.device('cuda:0')):
    logger.debug(f"Running Training Epoch {epoch}")

    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    scaler = GradScaler()
    total_loss = 0.

    for iter, data in enumerate(tqdm(data_loader)):
        mask, kspace, target, maximum, _, _, = data
        mask = mask.to(device)
        kspace = kspace.to(device)
        target = target.to(device)
        maximum = maximum.to(device)

        optimizer.zero_grad()
        if args.amp:
            with autocast():
                output = model(kspace, mask)
                if args.loss_mask :
                    output, target = output*(target>1e-5).float(), target*(target>1e-5).float()
                loss = loss_type(output, target, maximum) / args.grad_accmulation
            scaler.scale(loss).backward()

        else:
            output = model(kspace, mask)
            if args.loss_mask:
                output, target = output*(target>1e-5).float(), target*(target>1e-5).float()
            loss = loss_type(output, target, maximum) / args.grad_accmulation
            loss.backward()

        if ((iter + 1) % args.grad_accumulation) == 0:
            if args.grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if args.scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

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

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            mask, kspace, target, _, fnames, slices = data
            mask = mask.to(device)
            kspace = kspace.to(device)
            target = target.to(device)

            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

            if (iter % args.report_interval == 0 and (iter > 0)):
                print(f"{iter} validation done")

    for fname in reconstructions:
        reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname].items())])
    for fname in targets:
        targets[fname] = np.stack([out for _, out in sorted(targets[fname].items())])
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
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
    torch.cuda.set_device(device)
    logger.info("Current cuda device: %d", torch.cuda.current_device())

    # Choose Model
    if args.model == 'varnet':
        logger.info("model: varnet")
        model = VarNet(num_cascades=args.cascade,
                       chans=args.chans,
                       sens_chans=args.sens_chans,
                       unet=args.unet,
                       config=args.config)

    elif args.model == 'eamri':
        logger.info("model: eamri")
        model = EAMRI(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1, 3, 3, 3, 3],
                      fNums=[48, 96, 96, 96, 96], n_MSRB=3, shift=True)
    else:
        raise NotImplementedError("model not found")

    print_model_num_parameters(model)

    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # get scheduler
    if args.scheduler is not None:
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        else:
            raise NotImplementedError("scheduler not found")
    else:
        scheduler = None

    # get loss function
    if args.loss == "ssim":
        loss_type = SSIMLoss().to(device=device)
    elif args.loss == "mse":
        loss_type = torch.nn.MSELoss().to(device=device)
    elif args.loss == "edge":
        loss_type = EdgeMAELoss().to(device=device)
    else:
        raise NotImplementedError("loss not found")

    best_val_loss = 1.
    start_epoch = 0
    val_loss_log = np.empty((0, 2))

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True, aug=args.aug,
                                       edge=args.edge)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, shuffle=False, aug=False, edge=args.edge)

    # test saving and validation
    save_model(args, args.exp_dir, 0, model, optimizer, best_val_loss, False)
    val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

    for epoch in range(start_epoch, args.num_epochs):

        logger.info(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        logger.info(f"Actual Batch size: {args.batch_size} * {args.gradient_accumulation} = "
                    f"{args.batch_size * args.gradient_accumulation}")

        # train
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, scheduler, loss_type, device)

        # validate
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, device)

        # save loss
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        logger.info(f"loss file saved! {file_path}")

        # cal loss to tensor
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)
        val_loss = val_loss / num_subjects
        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        # save model
        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        logger.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        # save reconstructions if new best
        if is_new_best:
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            logger.info(f'ForwardTime = {time.perf_counter() - start:.4f}s')


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
