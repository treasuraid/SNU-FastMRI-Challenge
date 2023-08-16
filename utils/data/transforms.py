import random

import numpy as np
import torch
import cv2
import os, sys
from math import exp
from utils.model.fastmri.data.transforms import MaskFunc, apply_mask
from utils.model.fastmri.data.subsample import create_mask_for_mask_type
import torchvision.transforms.functional as TF

from utils.model.fastmri.data import transforms as T
from utils.model.fastmri import fft2c, ifft2c, rss_complex, complex_abs

from typing import List, Optional, Union

from torchvision import transforms as transforms
from torchvision.transforms import functional as TF
from utils.data.helper import *

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class MultiDataTransform2nd:
    def __init__(self, isforward, max_key, edge = False, aug = False):
        self.isforward = isforward
        self.max_key = max_key
        self.edge= edge 
        self.aug = aug
        
        if self.aug : 
        
            self.augmentation = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5), 
                 transforms.RandomVerticalFlip(p= 0.5), 
                 transforms.RandomAffine(degrees= 10, scale=(0.9, 1.1), shear= (-5,5,-5,5))])
            
            # mixup augmentation
    def __call__(self, recon_image, target_image, attrs, fname, num_slice) : 
        
        if not self.isforward: 
            targets = to_tensor(target_image)
            maximum = attrs[self.max_key]
        else : 
            targets = torch.Tensor([-1]*len(target_image))
            maximum = torch.Tensor([-1]*len(target_image))

        recons = to_tensor(recon_image) 
        
        if self.aug : 
            brightness = random.uniform(1.0, 2.0)
            targets = self.augmentation(targets * brightness) 
            recons = self.augmentation(recons * brightness) 
            maximum = maximum * brightness
        return recons, targets, maximum, fname, num_slice, brightness
        


# Transform for training image domain network
class DataTransform2nd:
    def __init__(self, isforward, max_key, edge = False, aug = False):
        self.isforward = isforward
        self.max_key = max_key
        self.edge= edge 
        self.aug = aug
        
        if self.aug : 
            
            self.augmentation = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5), 
                 transforms.RandomVerticalFlip(p= 0.5), 
                 transforms.RandomAffine(degrees= 10, translate =(0.1, 0.1), scale=(0.9, 1.1), shear= (-15,15,-15,15))])
            
            # mixup augmentation
            
        
            
    def __call__(self, input_image, grappa_image, recon_image, target_image, attrs, fname, slice) : 
        
        if not self.isforward: 
            target = to_tensor(target_image).unsqueeze(0) 
            maximum = attrs[self.max_key]
        else : 
            target = -1
            maximum = -1
            
        # numpy ndarray to torch tensor and stack input_image, grappa_image, target_image
        
        input_image = to_tensor(input_image).unsqueeze(0)  
        grappa_image = to_tensor(grappa_image).unsqueeze(0) 
        recon_image = to_tensor(recon_image).unsqueeze(0) 
        inputs = torch.cat((recon_image, grappa_image), dim = 0) # channel first 
        brightness = 1.0
        if self.aug : 
            brightness = random.uniform(1.0, 2.0)
            aug_input = torch.cat((input_image, recon_image, grappa_image, target), dim = 0)
            aug_output = self.augmentation(aug_input * brightness)
            inputs = aug_output[1:3,...] 
            target = aug_output[3,...]
        
        
        
        return inputs, target, maximum, fname, slice, brightness
        
        # add augmentation code for input images
         
        
        
        
    

class DataTransform:
    def __init__(self, isforward, max_key, edge=False, aug=False):
        self.isforward = isforward
        self.max_key = max_key
        self.edge = edge
        self.aug = aug
        # whether to use new mask algorithm
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target) # full image of 384 x 384
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        # todo add augmentation code for input kspace or image
        if self.aug :
            # mix mask with cartesian mask (1, 1, W, 1) -> (1, 1, W, 1)
            mask = np.roll(mask, random.randint(-2, 2), axis=-2)

        masked_kspace = to_tensor(input * mask)
        origin_kspace = to_tensor(input)
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=-1)
        
        mask = torch.from_numpy(mask.reshape(1, 1, masked_kspace.shape[-2], 1).astype(np.float32)).byte()

        if self.edge:
            return mask, masked_kspace, origin_kspace, target, getSobel(target), maximum, fname, slice
        else :
            return mask, masked_kspace, origin_kspace, target, -1, maximum, fname, slice


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models with added MRAugment data augmentation.
    """

    def __init__(self, augmentor=None, use_seed: bool = True):
        """
        Args:
            augmentor: DataAugmentor object that encompasses the MRAugment pipeline and
                schedules the augmentation probability
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        print("use_varnet_transform")

        self.use_seed = use_seed
        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False

    def __call__(self, mask, kspace, target, attrs, fname, slice_num):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            tuple containing:
                masked_kspace: k-space after applying sampling mask.
                mask: The applied sampling mask
                target: The target image (if applicable).
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        # Make sure data types match
        kspace = kspace.astype(np.complex64)
        target = target.astype(np.float32)

        if target is not None:
            target = to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = to_tensor(kspace)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        # Apply augmentations if needed
        if self.use_augment:
            if self.augmentor.schedule_p() > -0.0001:
                
                kspace, target = self.augmentor(kspace, target.shape)

        
        mask = np.roll(mask, random.randint(-2, 2), axis=0)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).float()
        masked_kspace = kspace * mask + 0.0
        return (
            mask.float(),
            masked_kspace,
            -1,
            target,
            -1,
            max_value,
            fname,
            slice_num,
        )

    def seed_pipeline(self, seed):
        """
        Sets random seed for the MRAugment pipeline. It is important to provide
        different seed to different workers and across different GPUs to keep
        the augmentations diverse.

        For an example how to set it see worker_init in pl_modules/fastmri_data_module.py
        """
        if self.use_augment:
            if self.augmentor.aug_on:
                self.augmentor.augmentation_pipeline.rng.seed(seed)

def getSobel(target):
    target = target.numpy()
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(target, 3, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(target, 3, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


class NoMaskTransform(DataTransform) :

    def __init__(self, isforward, max_key, edge=False, aug=False):
        super().__init__(isforward, max_key, edge, aug)

    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target) # full image of 384 x 384
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        if self.aug :
            # mix mask with cartesian mask (1, 1, W, 1) -> (1, 1, W, 1)
            mask = np.roll(mask, random.randint(-2, 2), axis=-2)

        masked_kspace = to_tensor(input)
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=-1)
        mask = torch.from_numpy(np.ones((1, 1, masked_kspace.shape[-2], 1)).astype(np.float32)).byte()


        return mask, masked_kspace,  target, -1, maximum, fname, slice,




def RandomMaskFunc() :

    pass

def test_getSobel(image_path):

    target = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    grad =getSobel(target)

    # visualize
    cv2.imshow('image', target)
    cv2.imshow('grad', grad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dithering(image : np.ndarray) :

    pass


"""
MRAugment applies channel-by-channel random data augmentation to MRI slices.
For example usage on the fastMRI and Stanford MRI datasets check out the scripts
in mraugment_examples.
"""


class AugmentationPipeline:
    """
    Describes the transformations applied to MRI data and handles
    augmentation probabilities including generating random parameters for
    each augmentation.
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.weight_dict = {
            'translation': hparams.aug_weight_translation,
            'rotation': hparams.aug_weight_rotation,
            'scaling': hparams.aug_weight_scaling,
            'shearing': hparams.aug_weight_shearing,
            'rot90': hparams.aug_weight_rot90,
            'fliph': hparams.aug_weight_fliph,
            'flipv': hparams.aug_weight_flipv
        }
        self.upsample_augment = hparams.aug_upsample
        self.upsample_factor = hparams.aug_upsample_factor
        self.upsample_order = hparams.aug_upsample_order
        self.transform_order = hparams.aug_interpolation_order
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im, max_output_size=None):
        # Trailing dims must be image height and width (for torchvision)
        im = complex_channel_first(im)
        # ---------------------------
        # pixel preserving transforms
        # ---------------------------
        # Horizontal flip
        if self.random_apply('fliph'):
            im = TF.hflip(im)

        # Vertical flip
        if self.random_apply('flipv'):
            im = TF.vflip(im)

        # Rotation by multiples of 90 deg
        if self.random_apply('rot90'):
            k = self.rng.randint(1, 4)
            im = torch.rot90(im, k, dims=[-2, -1])

        # Translation by integer number of pixels
        if self.random_apply('translation'):
            h, w = im.shape[-2:]
            t_x = self.rng.uniform(-self.hparams.aug_max_translation_x, self.hparams.aug_max_translation_x)
            t_x = int(t_x * h)
            t_y = self.rng.uniform(-self.hparams.aug_max_translation_y, self.hparams.aug_max_translation_y)
            t_y = int(t_y * w)

            pad, top, left = self._get_translate_padding_and_crop(im, (t_x, t_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.crop(im, top, left, h, w)

        # ------------------------
        # interpolating transforms
        # ------------------------
        interp = False

        # Rotation
        if self.random_apply('rotation'):
            interp = True
            rot = self.rng.uniform(-self.hparams.aug_max_rotation, self.hparams.aug_max_rotation)
        else:
            rot = 0.

        # Shearing
        if self.random_apply('shearing'):
            interp = True
            shear_x = self.rng.uniform(-self.hparams.aug_max_shearing_x, self.hparams.aug_max_shearing_x)
            shear_y = self.rng.uniform(-self.hparams.aug_max_shearing_y, self.hparams.aug_max_shearing_y)
        else:
            shear_x, shear_y = 0., 0.

        # Scaling
        if self.random_apply('scaling'):
            interp = True
            scale = self.rng.uniform(1 - self.hparams.aug_max_scaling, 1 + self.hparams.aug_max_scaling)
        else:
            scale = 1.

        # Upsample if needed
        upsample = interp and self.upsample_augment
        if upsample:
            upsampled_shape = [im.shape[-2] * self.upsample_factor, im.shape[-1] * self.upsample_factor]
            original_shape = im.shape[-2:]
            interpolation = TF.InterpolationMode.BICUBIC if self.upsample_order == 3 else TF.InterpolationMode.BILINEAR
            im = TF.resize(im, size=upsampled_shape, interpolation=interpolation)

        # Apply interpolating transformations
        # Affine transform - if any of the affine transforms is randomly picked
        if interp:
            h, w = im.shape[-2:]
            pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.affine(im,
                           angle=rot,
                           scale=scale,
                           shear=(shear_x, shear_y),
                           translate=[0, 0],
                           interpolation=TF.InterpolationMode.BILINEAR
                           )
            im = TF.center_crop(im, (h, w))

        # ---------------------------------------------------------------------
        # Apply additional interpolating augmentations here before downsampling
        # ---------------------------------------------------------------------

        # Downsampling
        if upsample:
            im = TF.resize(im, size=original_shape, interpolation=interpolation)

        # Final cropping if augmented image is too large
        if max_output_size is not None:
            im = crop_if_needed(im, max_output_size)

        # Reset original channel ordering
        im = complex_channel_last(im)

        return im

    def augment_from_kspace(self, kspace, target_size, max_train_size=None):
        im = ifft2c(kspace)
        
        # roll image to make brain center
        # replace np.roll(np.roll(np.fft.ifft2(kspace), shift=h // 2, axis=1), shift=w // 2, axis=2) with tensor operations
                
        im = self.augment_image(im, max_output_size=max_train_size)
        # print(im.shape, target_size)
        target = self.im_to_target(im, target_size)
         
        kspace = fft2c(im)
        return kspace, target

    def im_to_target(self, im, target_size):
        # Make sure target fits in the augmented image
        cropped_size = [min(im.shape[-3], target_size[0]),
                        min(im.shape[-2], target_size[1])]

        if len(im.shape) == 3:
            # Single-coil
            target = complex_abs(T.complex_center_crop(im, cropped_size))
        else:
            # Multi-coil
            assert len(im.shape) == 4
            target = T.center_crop(rss_complex(im), cropped_size)
        return target

    def random_apply(self, transform_name):
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else:
            return False

    def set_augmentation_strength(self, p):
        self.augmentation_strength = p

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """
        Calculates the necessary padding size before applying the
        general affine transformation. The output image size is determined based on the
        input image size and the affine transformation matrix.
        """
        h, w = im.shape[-2:]
        corners = [
            [-h / 2, -w / 2, 1.],
            [-h / 2, w / 2, 1.],
            [h / 2, w / 2, 1.],
            [h / 2, -w / 2, 1.]
        ]
        mx = torch.tensor(
            TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2, 3)
        corners = torch.cat([torch.tensor(c).reshape(3, 1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h - 1)
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2), min=0.0, max=w - 1)
        return int(py.item()), int(px.item())

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        t_x, t_y = translation
        h, w = im.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1)  # pad bottom
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1)  # pad top
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1)  # pad left
            left = 0
        else:
            pad[2] = min(-t_y, w - 1)  # pad right
            left = pad[2]
        return pad, top, left


class DataAugmentor:
    """
    High-level class encompassing the augmentation pipeline and augmentation
    probability scheduling. A DataAugmentor instance can be initialized in the
    main training code and passed to the DataTransform to be applied
    to the training data.
    """

    def __init__(self, hparams):
        """
        hparams: refer to the arguments below in add_augmentation_specific_args
        and is used to schedule the augmentation probability.
        """
        self.current_epoch = 0
        self.hparams = hparams
        self.aug_on = hparams.aug
        if self.aug_on:
            self.augmentation_pipeline = AugmentationPipeline(hparams)
        self.max_train_resolution = hparams.max_train_resolution

    def __call__(self, kspace, target_size):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [C, H, W, 2] (multi-coil) or [H, W, 2]
            where last dim is for real/imaginary channels
        target_size: [H, W] shape of the generated augmented target
        """
        # Set augmentation probability
        if self.aug_on:
            p = self.schedule_p()
            self.augmentation_pipeline.set_augmentation_strength(p)
        else:
            p = 0.0
        # Augment if needed
        if self.aug_on and p > -0.00001:
            # print("augmenting")
            kspace, target = self.augmentation_pipeline.augment_from_kspace(kspace,
                                                                            target_size=target_size,
                                                                            max_train_size=self.max_train_resolution)
        else:
            # Crop in image space if image is too large
            if self.max_train_resolution is not None:
                if kspace.shape[-3] > self.max_train_resolution[0] or kspace.shape[-2] > self.max_train_resolution[1]:
                    im = ifft2c(kspace)
                    im = complex_crop_if_needed(im, self.max_train_resolution)
                    kspace = fft2c(im)

        return kspace, target

    def schedule_p(self):
        D = self.hparams.aug_delay
        T = self.hparams.max_epochs
        t = self.current_epoch
        p_max = self.hparams.aug_strength

        if t < D:
            return 0.0
        else:
            if self.hparams.aug_schedule == 'constant':
                p = p_max
            elif self.hparams.aug_schedule == 'ramp':
                p = (t - D) / (T - D) * p_max
            elif self.hparams.aug_schedule == 'exp':
                c = self.hparams.aug_exp_decay / (T - D)  # Decay coefficient
                p = p_max / (1 - exp(-(T - D) * c)) * (1 - exp(-(t - D) * c))
            return p

    def add_augmentation_specific_args(parser):
        # parser.add_argument(
        #     '--aug_on',
        #     default=True,
        #     help='This switch turns data augmentation on.',
        #     action='store_true'
        # )
        # --------------------------------------------
        # Related to augmentation strenght scheduling
        # --------------------------------------------
        parser.add_argument(
            '--aug_schedule',
            type=str,
            default='exp',
            help='Type of data augmentation strength scheduling. Options: constant, ramp, exp'
        )
        parser.add_argument(
            '--aug_delay',
            type=int,
            default=17,
            help='Number of epochs at the beginning of training without data augmentation. The schedule in --aug_schedule will be adjusted so that at the last epoch the augmentation strength is --aug_strength.'
        )
        parser.add_argument(
            '--aug_strength',
            type=float,
            default=1.0,
            help='Augmentation strength, combined with --aug_schedule determines the augmentation strength in each epoch'
        )
        parser.add_argument(
            '--aug_exp_decay',
            type=float,
            default=5.0,
            help='Exponential decay coefficient if --aug_schedule is set to exp. 1.0 is close to linear, 10.0 is close to step function'
        )

        # --------------------------------------------
        # Related to interpolation
        # --------------------------------------------
        parser.add_argument(
            '--aug_interpolation_order',
            type=int,
            default=1,
            help='Order of interpolation filter used in data augmentation, 1: bilinear, 3:bicubic. Bicubic is not supported yet.'
        )
        parser.add_argument(
            '--aug_upsample',
            default=False,
            action='store_true',
            help='Set to upsample before augmentation to avoid aliasing artifacts. Adds heavy extra computation.',
        )
        parser.add_argument(
            '--aug_upsample_factor',
            type=int,
            default=2,
            help='Factor of upsampling before augmentation, if --aug_upsample is set'
        )
        parser.add_argument(
            '--aug_upsample_order',
            type=int,
            default=1,
            help='Order of upsampling filter before augmentation, 1: bilinear, 3:bicubic'
        )

        # --------------------------------------------
        # Related to transformation probability weights
        # --------------------------------------------
        parser.add_argument(
            '--aug_weight_translation',
            type=float,
            default=1.0,
            help='Weight of translation probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_rotation',
            type=float,
            default=1.0,
            help='Weight of arbitrary rotation probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_shearing',
            type=float,
            default=1.0,
            help='Weight of shearing probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_scaling',
            type=float,
            default=1.0,
            help='Weight of scaling probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_rot90',
            type=float,
            default=0.0,
            help='Weight of probability of rotation by multiples of 90 degrees. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_fliph',
            type=float,
            default=1.0,
            help='Weight of horizontal flip probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_flipv',
            type=float,
            default=1.0,
            help='Weight of vertical flip probability. Augmentation probability will be multiplied by this constant'
        )

        # --------------------------------------------
        # Related to transformation limits
        # --------------------------------------------
        parser.add_argument(
            '--aug_max_translation_x',
            type=float,
            default=0.06,
            help='Maximum translation applied along the x axis as fraction of image width'
        )
        parser.add_argument(
            '--aug_max_translation_y',
            type=float,
            default=0.06,
            help='Maximum translation applied along the y axis as fraction of image height'
        )
        parser.add_argument(
            '--aug_max_rotation',
            type=float,
            default=10.0,
            help='Maximum rotation applied in either clockwise or counter-clockwise direction in degrees.'
        )
        parser.add_argument(
            '--aug_max_shearing_x',
            type=float,
            default=5.0,
            help='Maximum shearing applied in either positive or negative direction in degrees along x axis.'
        )
        parser.add_argument(
            '--aug_max_shearing_y',
            type=float,
            default=5.0,
            help='Maximum shearing applied in either positive or negative direction in degrees along y axis.'
        )
        parser.add_argument(
            '--aug_max_scaling',
            type=float,
            default=0.25,
            help='Maximum scaling applied as fraction of image dimensions. If set to s, a scaling factor between 1.0-s and 1.0+s will be applied.'
        )

        # ---------------------------------------------------
        # Additional arguments not specific to augmentations
        # ---------------------------------------------------
        parser.add_argument(
            "--max_train_resolution",
            nargs="+",
            default=None,
            type=int,
            help="If given, training slices will be center cropped to this size if larger along any dimension.",
        )
        return parser

if __name__ == "__main__" :
    #
    # image_path = "./sobel_test.png"
    # test_getSobel(image_path)
    import h5py
    kspace_fname = "./brain_acc4_137.h5"
    with h5py.File(kspace_fname, "r") as hf:

        input = hf["kspace"][0]
        mask = np.array(hf["mask"])

