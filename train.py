from argparse import ArgumentParser

import torch
import argparse
import shutil
import os, sys
from pathlib import Path
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix
from utils.data.transforms import DataAugmentor
import logging
from logging import getLogger
logger = getLogger(__name__)

import wandb ## for logging


def parse():

    parser: ArgumentParser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network', required=True)
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=12, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=18, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=8, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    parser.add_argument('--seed', type=int, default=430, help='Fix random seed', required=True)

    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")


    # train loss mask
    parser.add_argument("--loss_mask", default=False, action = "store_true", help="Use mask for training Loss")

    # model
    parser.add_argument('--model', type=str, default='varnet', choices = ["varnet", "eamri"], help='Model to train')


    # gradient
    parser.add_argument('--grad_accumulation', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--grad_norm', type=float, default=1e8, help='Gradient clipping')
    parser.add_argument('--amp', type= bool, default = False, help='Use automatic mixed precision training')

    parser.add_argument('--unet', type= str, default = "plain", choices = ["plain", "swin"])
    parser.add_argument('--config', type=str, default = "./utils/model/config/swin_36.yaml", help = "config of swinUnetblock")

    # scheduler
    parser.add_argument('--scheduler', type=str, default=None, choices = [None, "cosine", "step"], help='Scheduler to train')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for scheduler')


    # loss
    parser.add_argument('--loss', type=str, default='ssim', choices = ["mse", "ssim", "edge"], help='Loss to train')
    parser.add_argument("--edge_weight", type=float, default=1, help="Weight for edge loss") # 1 in original EAMRI paper

    # data
    parser.add_argument("--aug", default=False, action = "store_true", help="Use augmentation for training")
    parser.add_argument("--edge", default=False, action = "store_true", help="Use edge image for training and validation")
    parser.add_argument("--collate", default=False, action = "store_true", help="Use collate function for training and validation")

    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    # augmentation aug delay max epochs aug strength
    # parser.add_argument("--aug_delay", type=int, default=0, help="Augmentation delay")
    # parser.add_argument("--max_epochs", type=int, default=30, help="Max epochs for augmentation")
    # parser.add_argument("--aug_strength", type=float, default=0.5, help="Augmentation strength")
    # parser.add_argument("--aug_schedule", type=str, default="ramp", choices = ["constant", "ramp", "exponent"], help="Augmentation schedule")

    DataAugmentor.add_augmentation_specific_args(parser)


    args = parser.parse_args()
    # todo : argument to config file (yaml) for better readability
    return args

if __name__ == '__main__':

    args = parse()
    logger.setLevel(logging.DEBUG)
    logginghandler = logging.StreamHandler(sys.stdout)
    logginghandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(logginghandler)
    logger.warning(args)


    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)
    
    args.amp = False 
    
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="fast-mri", config=vars(args))

    train(args)
