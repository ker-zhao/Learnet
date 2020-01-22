import numpy as np


def normal(shape):
    scale = 0.1
    return np.random.randn(*shape) * scale


def zeros(shape):
    return np.zeros(shape)
