from learnet import core as ty
from learnet import initializers
from learnet import nn


acts = {
    "relu": nn.relu,
    "sigmoid": nn.sigmoid,
    "softmax": nn.softmax
}


class Layer(object):
    def __init__(self, output_units=None, input_dims=None, kernel_regularizer=None):
        self.output_units = output_units
        self.input_dims = input_dims
        self.kernel_regularizer = kernel_regularizer


class Dense(Layer):
    def __init__(self, output_units, input_dims=None, activation=None, kernel_regularizer=None):
        super().__init__(output_units, input_dims, kernel_regularizer)
        self.output_units = output_units
        self.input_dims = input_dims
        self.activation = acts[activation] if isinstance(activation, str) else activation
        self.kernel_regularizer = kernel_regularizer
        self.w = None
        self.b = None

    def get_graph(self, inp, input_dims):
        self.input_dims = input_dims
        self.w = ty.get_variable(shape=(self.input_dims, self.output_units), initializer=initializers.normal, name="w")
        self.b = ty.get_variable(shape=(1, self.output_units), initializer=initializers.zeros, name="b")
        _z = ty.matmul(inp, self.w)
        z = ty.add(_z, ty.broadcast(self.b, _z))
        a = self.activation(z) if self.activation else z
        return a


class Dropout(Layer):
    def __init__(self, drop_rate, input_dims=None):
        super().__init__(output_units=None, input_dims=input_dims)
        self.rate = drop_rate

    def get_graph(self, inp, input_dims):
        self.output_units = input_dims
        return ty.dropout(inp, self.rate)
