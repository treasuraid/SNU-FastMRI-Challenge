import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part import forward

    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet_12_gracheck2', help='Name of network')
    parser.add_argument('-p', '--data_path', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    parser.add_argument('-m', '--mask', type=str, default='acc4', choices=['acc4', 'acc8'], help='type of mask | acc4 or acc8')
    
    parser.add_argument('--cascade', type=int, default=12, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=18, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=8, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    parser.add_argument("--edge", type=bool, default = False)
    parser.add_argument("--aug", type=bool, default= False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--collate", type=bool, default=False)
    parser.add_argument("--mode", type=str, choices= ["leaderboard", "train", "val"], required=True)
    parser.add_argument("--ckpt_name", type=str, default= "best_model.pt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.edge = False
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    if args.mode == "leaderboard":
        args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / args.mask
        args.data_path = args.data_path / args.mask
    elif args.mode == "val": 
        args.forward_dir = "../result" / args.net_name / "reconstructions_val"
    
    else : 
        args.forward_dir = "../result" / args.net_name / "reconstructions_train"
        
    
    print(args.forward_dir, args.data_path)
    
    forward(args)
