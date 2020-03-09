import gzip
import pickle
import os
from learnet import lib
from learnet import nn


def load_data():
    script_dir = os.path.dirname(__file__)
    mnist_file = os.path.join(os.path.join(script_dir, 'data'), 'mnist.pkl.gz')

    with gzip.open(mnist_file, 'rb') as mnist_file:
        u = pickle._Unpickler(mnist_file)
        u.encoding = 'latin1'
        train, val, test = u.load()
    # train, val, test = [train[0].T, train[1].T], [val[0].T, val[1].T], [test[0].T, test[1].T]
    if lib.GPU_ENABLED:
        train = (lib.cp.asarray(train[0]), lib.cp.asarray(train[1]))
        val = (lib.cp.asarray(val[0]), lib.cp.asarray(val[1]))
        test = (lib.cp.asarray(test[0]), lib.cp.asarray(test[1]))
    train = (train[0], nn.one_hot(train[1], 10))
    val = (val[0], nn.one_hot(val[1], 10))
    test = (test[0], nn.one_hot(test[1], 10))
    return train, val, test
