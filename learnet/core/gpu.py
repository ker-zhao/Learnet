from numba import cuda
from numba import 
import math


@cuda.jit
def _add(a, b, result):
    x, y = cuda.grid(2)
    if x < result.shape[0] and y < result.shape[1]:
        result[x, y] = a[x, y] + b[x, y]


def add(a, b):
    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    gpu_result = cuda.device_array_like(a)
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(math.ceil(a.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(b.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    _add[blocks_per_grid, threads_per_block](a_device, b_device, gpu_result)
    result = gpu_result.copy_to_host()
    return result
