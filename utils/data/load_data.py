import h5py
import random
from utils.data.transforms import DataTransform, VarNetDataTransform, DataAugmentor
from utils.model.fastmri.data.subsample import create_mask_for_mask_type
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 
from pathlib import Path
import numpy as np

import os

class SliceData2nd(Dataset):
    
    def __init__(self, input_root, recon_root, transform, input_key, 
                 target_key, forward=False, edge=False):
        self.transform = transform
        print(input_key)
        self.input_key = input_key
        self.target_key = target_key 
        self.forward = forward 
        self.edge = edge
        self.input_root = input_root 
        self.recon_root = recon_root
        
        self.input_examples = []
        
        image_files = os.listdir(input_root)
        
        for fname in sorted(image_files):
            num_slices = self._get_metadata(os.path.join(input_root, fname))
            self.input_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
            
    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, idx):
        input_fname, dataslice = self.input_examples[idx]

        # filename is identical for input, grappa, recon

        # load three images

        with h5py.File(os.path.join(self.input_root,input_fname), 'r') as f:
#             print(f.keys())
            input_image = np.array(f[self.input_key])[dataslice].astype(np.float32)
            target_image = np.array(f[self.target_key])[dataslice].astype(np.float32)
            grappa_image = np.array(f["image_grappa"])[dataslice].astype(np.float32)
            attr = dict(f.attrs)

        with h5py.File(os.path.join(self.recon_root,input_fname), 'r') as f:
            recon_image = np.array(f["reconstruction"])[dataslice].astype(np.float32)

        return self.transform(input_image, grappa_image, recon_image, target_image, attr, input_fname, dataslice)

class MultiSliceData2nd(Dataset):
    def __init__(self, input_root, recon_root, transform, input_key, 
                 target_key, forward=False, edge=False, num_slices=3): 
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key 
        self.forward = forward 
        self.edge = edge
        self.input_root = input_root 
        self.recon_root = recon_root
        self.num_slices = num_slices
        
        self.input_examples = []
        self.sampling_weights = []
        image_files = os.listdir(input_root)
        
        for fname in sorted(image_files):
            num_slices = self._get_metadata(os.path.join(input_root, fname))
            self.input_examples += [(fname, slice_ind) for slice_ind in range(num_slices-self.num_slices+1)]
            self.sampling_weights += [(1.1) - (0.2)*i/(num_slices -self.num_slices)for i in range(num_slices -self.num_slices +1)]
            
    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, idx):
        input_fname, dataslice = self.input_examples[idx]

        # filename is identical for input, grappa, recon
        # load three images
        with h5py.File(os.path.join(self.input_root,input_fname), 'r') as f:
            target_image = np.array(f[self.target_key])[dataslice:dataslice+self.num_slices].astype(np.float32) # 3*384*384 
            attr = dict(f.attrs)

        with h5py.File(os.path.join(self.recon_root,input_fname), 'r') as f:
            recon_image = np.array(f["reconstruction"])[dataslice:dataslice+self.num_slices].astype(np.float32) # 3*384*384 

        return self.transform(recon_image, target_image, attr, input_fname, dataslice)
    
    


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, edge=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.weights = []
        self.edge = edge
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                self.image_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)
            self.kspace_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
            # 2 times more weight for 1 slice than last slice and sum of weights in each fname is 1
            # print([(1.25) - (0.5)*i/(num_slices -1) for i in range(num_slices)], sum([(1.25) - (0.5)*i/(num_slices -1) for i in range(num_slices)]), num_slices)
            self.weights += [(1.4) - (0.8)*i/(num_slices -1) for i in range(num_slices)]
        self.weights = np.array(self.weights)      

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        """
        Args:
            i (int): Index
        Returns:
            mask: [B, 1, 1, W, 1]
            masked_kspace: [B, Num_Coils, H, W, 2]
            target: [B, 1, 384, 384] # full image of 384 x 384 or tuple([B, 1, 384, 384], [B, 1, 384, 384]) # full image of 384 x 384 and edge image of 384 x 384
            maximum: [B, 1]
            fname: [B, 1]
            slice: [B, 1]
        """
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])

        if not self.forward:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        else:
            target = -1
            attrs = -1



        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False, aug= False, mode = 'train'):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    if aug == False  :
        data_storage = SliceData(
            root=data_path,
            transform=DataTransform(isforward, max_key_, args.edge, False),
            input_key=args.input_key,
            target_key=target_key_,
            forward = isforward,
            edge= args.edge
        )

        data_loader = DataLoader(
            dataset=data_storage,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_fn if args.collate else None,
        )
    else :
        augmentor = DataAugmentor(args)
        # #mask = create_mask_for_mask_type(
        #     "equispaced", 0.08
        # )
            
        data_storage = SliceData(
            root=data_path,
            transform=VarNetDataTransform(augmentor, use_seed=False),
            input_key=args.input_key,
            target_key=target_key_,
            forward=isforward,
            edge=args.edge,
        )
        if args.wrs:
            print("Using Weighted Random Sampler")
            data_loader = DataLoader(
                dataset=data_storage,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=collate_fn if args.collate else None,
                sampler=WeightedRandomSampler(data_storage.weights, len(data_storage.weights)),
            )
        else :
            data_loader = DataLoader(
                dataset=data_storage,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                collate_fn=collate_fn if args.collate else None,
            )

    return data_storage, data_loader


def collate_fn(batch):
    # todo : implement collate_fn
    # batch is a list of dictionaries
    # each dictionary is the output of DataTransform

    # stack simply targets, edge_targets, maximum, fnames and slices
    target = np.stack([b[2] for b in batch], axis=0)
    edge = np.stack([b[3] for b in batch], axis=0)
    maximum = np.stack([b[4] for b in batch], axis=0)
    fname = [b[5] for b in batch]
    slice = [b[6] for b in batch]

    pass


class ReconData(Dataset) : 
    
    def __init__(self, root, root_recon, transform, input_key, target_key, forward=False, edge=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.edge = edge
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                self.image_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)
            self.kspace_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices
    
    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        """
        Args:
            i (int): Index
        Returns:
            mask: [B, 1, 1, W, 1]
            masked_kspace: [B, Num_Coils, H, W, 2]
            target: [B, 1, 384, 384] # full image of 384 x 384 or tuple([B, 1, 384, 384], [B, 1, 384, 384]) # full image of 384 x 384 and edge image of 384 x 384
            maximum: [B, 1]
            fname: [B, 1]
            slice: [B, 1]
        """
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])

        if not self.forward:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        else:
            target = -1
            attrs = -1



        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)

