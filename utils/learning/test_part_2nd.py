import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet


from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet
from utils.model.kbnet import KBNet_l,  KBNet_s
from utils.model.NAFNet_arch import NAFNet
from utils.data.load_data import SliceData2nd, MultiSliceData2nd
from utils.data.transforms import DataTransform2nd, MultiDataTransform2nd
import pandas as pd 
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from kornia.morphology import dilation, erosion 

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    with torch.no_grad():
        for data in data_loader:
            input, target, maximum, fnames, slices, brightness = data 
            
            input = input.cuda(non_blocking=True)
            output = model(input).squeeze(dim=0)
            print(output.shape)
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())


    if args.model == "kbnet":
        model = KBNet_s(img_channel=3, out_channel=1 if not args.multi_channel else 3, width=32, middle_blk_num=6, enc_blk_nums=[2, 2, 2, 2],
                    dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=True, ffn_scale=1.5).to(device=device)
    
    elif args.model == "nafnet":
        width = 32
        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]
        model = NAFNet(img_channel=args.input_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / args.ckpt_name , map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    print(args.data_path)
    test_transform = DataTransform2nd(isforward= False, max_key= args.max_key, edge = args.edge, aug = False)
    forward_loader = torch.utils.data.DataLoader(SliceData2nd(args.data_path, 
                                args.recon_path / "reconstructions_leaderboard"/ args.mask, 
                                transform=test_transform,
                                input_key = args.input_key,
                                target_key= args.target_key),
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers)
        
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)