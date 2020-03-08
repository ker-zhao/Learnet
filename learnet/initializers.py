from learnet import lib


def normal(shape):
    scale = 0.1
    return lib.np.random.randn(*shape) * scale


def zeros(shape):
    return lib.np.zeros(shape)
