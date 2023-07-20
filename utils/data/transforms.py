import random

import numpy as np
import torch
import cv2

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
            return mask, masked_kspace, origin_kspace, target, getSobel(target), maximum, fname, slice,
        else :
            return mask, masked_kspace, origin_kspace, target, -1, maximum, fname, slice,


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


if __name__ == "__main__" :
    #
    # image_path = "./sobel_test.png"
    # test_getSobel(image_path)
    import h5py
    kspace_fname = "./brain_acc4_137.h5"
    with h5py.File(kspace_fname, "r") as hf:

        input = hf["kspace"][0]
        mask = np.array(hf["mask"])

