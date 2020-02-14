from learnet import type as ty
from learnet import initializers


class Layer(object):
    pass


class Dense(Layer):
    def __init__(self, output_units, input_dims=None, activation=None):
        self.output_units = output_units
        self.activation = activation
        self.input_dims = input_dims

    def get_graph(self, inp, input_dims):
        self.input_dims = input_dims
        w = ty.get_variable(shape=(self.input_dims, self.output_units),
                            initializer=initializers.normal, name="w")
        b = ty.get_variable(shape=(1, self.output_units), initializer=initializers.zeros, name="b")
        _z = ty.matmul(inp, w)
        z = ty.add(_z, ty.broadcast(b, _z))
        if self.activation:
            a = self.activation(z)
        else:
            a = z
        return a
