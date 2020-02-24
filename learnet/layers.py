import numpy as np

from learnet import type as ty
from learnet import initializers
from learnet import nn


acts = {
    "relu": nn.relu,
    "sigmoid": nn.sigmoid,
    "softmax": nn.softmax
}


class Layer(object):
    pass


class Dense(Layer):
    def __init__(self, output_units, input_dims=None, activation=None):
        self.output_units = output_units
        self.input_dims = input_dims
        self.activation = acts[activation] if isinstance(activation, str) else activation

    def get_graph(self, inp, input_dims):
        self.input_dims = input_dims
        w = ty.get_variable(shape=(self.input_dims, self.output_units),
                            initializer=initializers.normal, name="w")
        b = ty.get_variable(shape=(1, self.output_units), initializer=initializers.zeros, name="b")
        _z = ty.matmul(inp, w)
        z = ty.add(_z, ty.broadcast(b, _z))
        a = self.activation(z) if self.activation else z
        return a


class Dropout(Layer):
    def __init__(self, drop_rate):
        self.rate = drop_rate

    def get_graph(self, inp, input_dims):
        # Todo: There needs a dropout operator.
        np.random.rand()