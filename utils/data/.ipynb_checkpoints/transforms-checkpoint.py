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
    def __init__(self, isforward, max_key, edge=False ):
        self.isforward = isforward
        self.max_key = max_key
        self.edge = edge
        # whether to use new mask algorithm
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target) # full image of 384 x 384
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        masked_kspace = to_tensor(input * mask)
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, masked_kspace.shape[-2], 1).astype(np.float32)).byte()

        if self.edge:
            return mask, masked_kspace, (target,  to_tensor(getSobel(target))), maximum, fname, slice,
        else :
            return mask, masked_kspace, target, maximum, fname, slice,


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
def RandomMaskFunc() :

    pass