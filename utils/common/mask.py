"""
mask algorithm
"""
import numpy as np

def gen_random_mask(shape, center_fractions, accelerations ):
    """Generate a mask of fully sampled k-space lines.
    Args:
        shape (tuple): The shape of the mask to generate.
        center_fractions (int): Fraction of low-frequency columns to be retained.
        accelerations (int): Amount of under-sampling to be performed; the
            higher the acceleration, the fewer columns will be retained.
    Returns:
        torch.Tensor: A mask of shape ``shape``.
    """
    num_cols = shape[-2]
    mask = np.zeros(num_cols, dtype=np.float32)
    num_low_freqs = int(round(num_cols * center_fractions))
    mask[:num_low_freqs] = 1

    # Determine number of columns to be under-sampled
    num_high_freqs = int(round(num_cols * accelerations))

    # Randomly choose which columns will be under-sampled
    mask[num_low_freqs:num_low_freqs + num_high_freqs] = np.random.choice(
        [0, 1], num_high_freqs, p=[0.8, 0.2])

    np.random.shuffle(mask)
    mask = mask.reshape([1, 1, 1, -1, 1])
    return mask

if __name__ == "__main__" :

    output = gen_mask((1, 1, 1, 640, 1), [0.1], [4])