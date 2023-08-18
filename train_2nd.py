from argparse import ArgumentParser

import torch
import argparse
import shutil
import os, sys
from pathlib import Path
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
    
from utils.learning.train_part_2nd import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix
from utils.data.transforms import DataAugmentor
import logging
from logging import getLogger
logger = getLogger(__name__)

import wandb ## for logging


def parse() :
    
    
    parser = ArgumentParser(description='Train Image Domain Network on FastMRI challenge Images')
    
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network', required=True)
    
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/image', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/image', help='Directory of validation data')
 
    
    # grappa and reconstruction
    parser.add_argument("--mode", type=str, default="train", help="train or leaderboard or val",choices=["train", "leaderboard", "val"]) 
    parser.add_argument('--recon-path', type=Path, default='../result/varnet_16_aug_resume_from_epoch12/', help='Directory of recon data')
    parser.add_argument("--grappa-path", type=Path, default = "../../Data/grappa/", help = "If none use original grappa")
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=2023, help='Fix random seed')
    
    parser.add_argument("--loss_mask", type= bool ,default = True)
    parser.add_argument("--weight-decay", type=float, default=0.00005, help="Weight decay")
    parser.add_argument("--num-workers", type= int, default = 4)
    parser.add_argument("--edge", type=bool, default=False)
    parser.add_argument("--aug", type=bool, default=True)
    
    parser.add_argument("--grad_accumulation", type = int, default = 8)
    
    parser.add_argument("--multi-channel", type = bool, default = False)
    
    parser.add_argument("--input_channel", type = int, default = 3) # original, grappa, recon from e2evarnet
    
    parser.add_argument("--model", type = str, default = "nafnet", choices = ["nafnet", "kbnet"]) 
    parser.add_argument("--grappa_path", type= Path, default = "../../Data/grappa/")
    parser.add_argument("--grad_norm", type = float, default = 100)
    args = parser.parse_args()
    
    return args 

if __name__ == '__main__':
    args = parse()
    seed_fix(args.seed)
    print(args)
    
    # todo : if gpu-num == -1 then use cpu

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project="fast-mri", config=vars(args))
    
    
    train(args)
    

