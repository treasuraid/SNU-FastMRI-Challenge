import h5py
import random
from utils.data.transforms import DataTransform, VarNetDataTransform, DataAugmentor
from utils.model.fastmri.data.subsample import create_mask_for_mask_type
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, edge=False):
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
