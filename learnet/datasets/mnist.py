import gzip
import pickle
import os


def load_data():
    script_dir = os.path.dirname(__file__)
    mnist_file = os.path.join(os.path.join(script_dir, 'data'), 'mnist.pkl.gz')

    with gzip.open(mnist_file, 'rb') as mnist_file:
        u = pickle._Unpickler(mnist_file)
        u.encoding = 'latin1'
        train, val, test = u.load()
    # train, val, test = [train[0].T, train[1].T], [val[0].T, val[1].T], [test[0].T, test[1].T]
    return train, val, test