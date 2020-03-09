__author__ = "Ker"

from . import lib


class Node(object):
    def __init__(self, _type, inputs=None, value=None):
        self.type = _type
        self.inputs = inputs
        self.value = value
        self.cache = None

    def erase_cache(self):
        self.cache = None
        if self.inputs:
            for node in self.inputs:
                node.erase_cache()

    def eval(self, feed_dict=None):
        if feed_dict is not None:
            for k, v in feed_dict.items():
                k.value = v
            self.erase_cache()
        if self.cache is not None:
            return self.cache
        else:
            if self.value is not None:
                self.cache = self.value
            else:
                self.cache = self.type.compute(self.inputs)
            return self.cache

    def diff(self, grad, grads_map):
        if self.inputs:
            grads = self.type.diff(self.inputs, grad)
            for i, in_node in enumerate(self.inputs):
                if in_node in grads_map:
                    grads_map[in_node] += grads[i]
                else:
                    grads_map[in_node] = grads[i]
                in_node.diff(grads[i], grads_map)
        else:
            return

    def gradients(self, nodes, feed_dict=None):
        self.eval(feed_dict=feed_dict)
        grads_map = {self: 1.}
        self.diff(1., grads_map)
        return [grads_map[i] for i in nodes]

    def get_variable_nodes(self, nodes=None):
        if nodes is None:
            nodes = []
        for node in self.inputs:
            if node.inputs:
                node.get_variable_nodes(nodes)
            elif isinstance(node.type, Variable):
                if node not in nodes:
                    nodes.append(node)
                else:
                    pass
            else:
                pass
        return nodes


class AddOp(object):
    @staticmethod
    def compute(inputs):
        return inputs[0].eval() + inputs[1].eval()

    @staticmethod
    def diff(_, grads):
        return [grads, grads]


def add(x, y):
    return Node(AddOp, [x, y])


class SubOp(object):
    @staticmethod
    def compute(inputs):
        return inputs[0].eval() - inputs[1].eval()

    @staticmethod
    def diff(_, grads):
        return [grads, -grads]


def sub(x, y):
    return Node(SubOp, [x, y])


class MulOp(object):
    @staticmethod
    def compute(inputs):
        return inputs[0].eval() * inputs[1].eval()

    @staticmethod
    def diff(inputs, grads):
        return [grads * inputs[1].cache, grads * inputs[0].cache]


def multiply(x, y):
    return Node(MulOp, [x, y])


class DivOp(object):
    @staticmethod
    def compute(inputs):
        return inputs[0].eval() / inputs[1].eval()

    @staticmethod
    def diff(inputs, grads):
        # left = grads / inputs[1].cache
        # right = -grads * inputs[0].cache / (inputs[1].cache * inputs[1].cache)
        # return [left, right]
        left = grads / inputs[1].cache
        a = (-grads) * inputs[0].cache
        b = inputs[1].cache * inputs[1].cache
        right = a / b
        return [left, right]


def div(x, y):
    return Node(DivOp, [x, y])


class MatmulOp(object):
    @staticmethod
    def compute(inputs):
        return lib.np.dot(inputs[0].eval(), inputs[1].eval())

    @staticmethod
    def diff(inputs, grads):
        left = lib.np.dot(grads, inputs[1].cache.T)
        right = lib.np.dot(inputs[0].cache.T, grads)
        return [left, right]


def matmul(x, y):
    return Node(MatmulOp, [x, y])


class ReduceSumOp(object):
    def __init__(self, axis):
        self.axis = axis

    def compute(self, inputs):
        return lib.np.sum(inputs[0].eval(), axis=self.axis, keepdims=True)

    @staticmethod
    def diff(inputs, grads):
        return [lib.np.ones(inputs[0].cache.shape) * grads]


def reduce_sum(x, axis=None):
    return Node(ReduceSumOp(axis), [x])


class ExpOp(object):
    @staticmethod
    def compute(inputs):
        return lib.np.exp(inputs[0].eval())

    @staticmethod
    def diff(inputs, grads):
        return [lib.np.exp(inputs[0].cache) * grads]


def exp(x):
    return Node(ExpOp, [x])


def replace_zeroes(data):
    data[data == 0] = 0.000000001
    return data


class LogOp(object):
    @staticmethod
    def compute(inputs):
        # inp = replace_zeroes(inputs[0].eval())
        # return lib.np.log(inp)
        return lib.np.log(inputs[0].eval())

    @staticmethod
    def diff(inputs, grads):
        return [grads / inputs[0].cache]


def log(x):
    return Node(LogOp, [x])


class MeanOp(object):
    def __init__(self, axis):
        self.axis = axis

    def compute(self, inputs):
        return lib.np.mean(inputs[0].eval(), axis=self.axis, keepdims=True)

    def diff(self, inputs, grads):
        m = inputs[0].cache.shape[self.axis] if self.axis is not None else \
            inputs[0].cache.shape[0] * inputs[0].cache.shape[1]
        return [lib.np.ones(inputs[0].cache.shape) * grads / m]


def mean(x, axis=None):
    return Node(MeanOp(axis), [x])


class ReluOp(object):
    @staticmethod
    def compute(inputs):
        return lib.np.maximum(inputs[0].eval(), 0)

    @staticmethod
    def diff(inputs, grads):
        return [lib.np.sign(lib.np.maximum(inputs[0].cache, 0)) * grads]


def relu(x):
    return Node(ReluOp, [x])


# Todo: This should not be an operator, because this can be combined.
class SigmoidOp(object):
    @staticmethod
    def compute(inputs):
        return 1 / (1 + lib.np.exp(-inputs[0].eval()))

    @staticmethod
    def diff(inputs, grads):
        return [lib.np.exp(inputs[0].cache) / (lib.np.exp(inputs[0].cache) + 1) ** 2 * grads]


def sigmoid(x):
    return Node(SigmoidOp, [x])


class TanhOp(object):
    @staticmethod
    def compute(inputs):
        x = inputs[0].eval()
        return (lib.np.exp(x) - lib.np.exp(-x)) / (lib.np.exp(x) + lib.np.exp(-x))

    @staticmethod
    def diff(inputs, grads):
        x = inputs[0].cache
        return [(1 - (lib.np.exp(x) - lib.np.exp(-x)) / (lib.np.exp(x) + lib.np.exp(-x)) ** 2) * grads]


def tanh(x):
    return Node(TanhOp, [x])


class NegativeOp(object):
    @staticmethod
    def compute(inputs):
        return -inputs[0].eval()

    @staticmethod
    def diff(_, grads):
        return [-grads]


def negative(x):
    return Node(NegativeOp, [x])


class DropoutOp(object):
    def __init__(self, rate):
        self.rate = rate
        self.d = None

    def compute(self, inputs):
        x = inputs[0].eval()
        self.d = lib.np.random.rand(x.shape[0], x.shape[1])
        self.d = self.d > self.rate
        dropped = lib.np.multiply(x, self.d)
        return dropped / (1. - self.rate)

    def diff(self, _, grads):
        return [lib.np.multiply(grads, self.d) / (1. - self.rate)]


def dropout(x, rate):
    return Node(DropoutOp(rate), [x])


class L2RegularizerOp(object):
    def __init__(self, lambd):
        self.lambd = lambd

    def compute(self, inputs):
        w = inputs[0].eval()
        x = inputs[1].eval()
        return lib.np.sum(lib.np.square(w)) * (self.lambd / (2 * x.shape[0]))

    def diff(self, inputs, _):
        w = inputs[0].cache
        x = inputs[1].cache
        return [w / x.shape[0] * self.lambd, 0]


def l2_regularizer(w, x, lambd=0.01):
    return Node(L2RegularizerOp(lambd), [w, x])


class Operator(object):
    pass


class Constant(object):
    def __init__(self, name):
        self.name = name
    pass


def constant(value, name="constant"):
    return Node(Constant(name), value=value)


class PlaceHolder(object):
    def __init__(self, name):
        self.name = name


def placeholder(name="placeholder"):
    return Node(PlaceHolder(name))


class Variable(object):
    def __init__(self, name):
        self.name = name


# Todo: Delete one of these two functions.
def variable(value, name="variable"):
    node = Node(Variable(name))
    node.value = value
    return node


def get_variable(shape, initializer, name="variable"):
    node = Node(Variable(name))
    node.value = initializer(shape)
    return node


class Optimizer(object):
    def __init__(self, opt):
        self.opt = opt

    def compute(self, _):
        return self.opt.step()


def optimizer(opt, cost):
    return Node(Optimizer(opt), [cost])


class BroadcastOp(object):
    @staticmethod
    def compute(inputs):
        return lib.np.broadcast_to(inputs[0].eval(), inputs[1].eval().shape)

    @staticmethod
    def diff(inputs, grads):
        if isinstance(inputs[0].cache, lib.np.ndarray):
            if inputs[0].cache.shape[0] == inputs[1].cache.shape[0]:
                axis = 1
            elif inputs[0].cache.shape[1] == inputs[1].cache.shape[1]:
                axis = 0
            else:
                axis = None
        else:
            axis = None
        # axis = 1 if inputs[0].cache.shape[0] == inputs[1].cache.shape[0] else 0
        left = lib.np.sum(grads, axis=axis, keepdims=True)
        right = lib.np.zeros(shape=inputs[1].cache.shape)
        return [left, right]


def broadcast(x, y):
    return Node(BroadcastOp, [x, y])
