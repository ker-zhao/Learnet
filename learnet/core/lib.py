import numpy
import cupy
np = numpy
cp = cupy
GPU_ENABLED = False


def enable_gpu(is_enabled):
    global GPU_ENABLED
    GPU_ENABLED = is_enabled
    if is_enabled:
        import cupy
        lib = cupy
    else:
        import numpy
        lib = numpy
    global np
    np = lib
