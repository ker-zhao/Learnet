import logging
import numpy as np
import learnet as ln


def run_tests():
    test_add()
    test_basic_1()
    test_basic_matrix_1()
    test_squared_error()
    test_softmax()
    test_mean()
    test_relu()
    test_sigmoid()
    test_optimizer()
    test_tanh()
    # test_model()


def test_add():
    v = ln.add(ln.add(ln.constant(1), ln.constant(2)), ln.constant(3)).eval()
    if v != 6:
        logging.error("test_add error, v should be 6, got v={}".format(v))


def test_basic_1():
    x = ln.constant(8)
    y = ln.constant(4)
    z = ln.multiply(ln.div(x, (ln.sub(ln.constant(8), y))), ln.constant(10))
    expect = [2.5, 5.0]
    result = z.gradients([x, y])
    if result != expect:
        logging.error("test_basic_1 error, v should be {}, got v={}".format(expect, result))


def test_basic_matrix_1():
    x = ln.constant(np.ones([3, 2]) * 8)
    y = ln.constant(np.ones([3, 2]) * 4)
    z = ln.multiply(ln.div(x, (ln.sub(ln.constant(8), y))), ln.constant(10))
    expect = 2.5
    result = z.gradients([x, y])[0][0, 0]
    if result != expect:
        logging.error("test_basic_matrix_1 error, v should be {}, got v={}".format(expect, result))


def test_mean():
    x = ln.constant(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))
    z = ln.mean(x)
    expect = 0.125
    result = z.gradients([x])[0][0, 0]
    if result != expect:
        logging.error("test_mean error, v should be {}, got v={}".format(expect, result))


def test_squared_error():
    x = ln.constant(np.ones([3, 10]) * 0.5)
    w1 = ln.constant(np.ones([2, 3]) * 4)
    z1 = ln.matmul(w1, x)
    w2 = ln.constant(np.ones([1, 2]) * 4)
    z2 = ln.matmul(w2, z1)
    y = ln.constant(24)
    cost = ln.div(ln.reduce_sum(ln.multiply(ln.multiply(ln.sub(z2, y), ln.sub(z2, y)), ln.constant(0.5))),
                  ln.constant(x.value.shape[1]))
    expect = 48.
    result = cost.gradients([w1])[0][0, 0]
    if abs(expect - result) > 0.0001:
        logging.error("test_squared_error error, v should be {}, got v={}".format(expect, result))


def test_softmax():
    epsilon = 1e-7
    logits = ln.variable(np.array([[1., 8.], [3., 8.]]))
    y = ln.constant(np.array([[0., 1.], [0., 1.]]))
    predict = ln.nn.softmax(logits)
    cost = ln.nn.cross_entropy(predict, y)
    logits_pos = ln.constant(np.array([[1, 4], [3, 8 + epsilon]]))
    logits_neg = ln.constant(np.array([[1, 4], [3, 8 - epsilon]]))
    softmax_pos = ln.nn.softmax(logits_pos)
    softmax_neg = ln.nn.softmax(logits_neg)
    result_pos = ln.nn.cross_entropy(softmax_pos, y)
    result_neg = ln.nn.cross_entropy(softmax_neg, y)
    num_grads = ln.div(
        ln.sub(result_pos, result_neg),
        ln.constant(epsilon * 2))
    expect = num_grads.eval()[0, 0]
    result = cost.gradients([logits])[0][1, 1]
    diff = abs(expect - result)
    if diff > epsilon:
        logging.error("test_softmax error, v should be {}, got v={}".format(expect, result))
    opt = ln.optimizers.GradientDescent(0.01)
    opt.minimize(cost)
    opt.gradient_check({})


def test_relu():
    x = ln.constant(np.array([[1, 2, 3, -4, 0], [-1, -2, -3, 4, 0]]))
    y = ln.relu(x)
    expect = np.array([[1., 1., 1., 0., 0.], [0., 0., 0., 1., 0.]])
    result = y.gradients([x])[0]
    if not np.array_equal(expect, result):
        logging.error("test_relu error, v should be {}, got v={}".format(expect, result))


def test_sigmoid():
    x = ln.constant(np.array([1, 0, -1]))
    y = ln.sigmoid(x)
    expect = 0.25
    result = y.gradients([x])[0][1]
    if not np.array_equal(expect, result):
        logging.error("test_relu error, v should be {}, got v={}".format(expect, result))


def test_tanh():
    x = ln.variable(np.array([0]))
    y = ln.tanh(x)
    expect = 1
    result = y.gradients([x])[0][0]
    if not np.array_equal(expect, result):
        logging.error("test_tanh error, v should be {}, got v={}".format(expect, result))


def test_optimizer():
    w = ln.variable(0)
    cons = ln.placeholder()
    cost = ln.add(ln.add(ln.multiply(w, w), ln.multiply(cons, w)), ln.constant(400))
    optimizer = ln.optimizers.GradientDescent().minimize(cost)
    for i in range(1000):
        optimizer.eval(feed_dict={cons: -40})
    expect = 20.
    result = w.eval()
    if abs(expect - result) > 0.00001:
        logging.error("test_optimizer error, v should be {}, got v={}".format(expect, result))


def accuracy(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)
    y = np.argmax(y, axis=1)
    correct = np.sum(np.equal(y_hat, y))
    return correct / y.shape[0] * 100.00


def test_model():
    (x_train, y_train), (x_val, y_val), _ = ln.datasets.mnist.load_data()
    m = 4096
    x_train, y_train = x_train[:m, :], y_train[:m]
    y_train = ln.nn.one_hot(y_train, 10)
    y_val = ln.nn.one_hot(y_val, 10)
    print("test_model, data's shape: ", x_train.shape, y_train.shape)

    # model = ln.models.Sequential([
    #     ln.layers.Dense(128, input_dims=x_train.shape[1], activation=ln.nn.relu),
    #     ln.layers.Dense(128, activation=ln.nn.relu),
    #     ln.layers.Dense(10, activation=ln.nn.softmax)
    # ])
    model = ln.models.Sequential()
    model.add(ln.layers.Dense(128, input_dims=x_train.shape[1], activation="relu"))
    model.add(ln.layers.Dense(128, activation="relu"))
    model.add(ln.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="Adam", loss=ln.nn.cross_entropy)
    # model.grad_check(x_train, y_train)
    model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1)
    model.evaluate(x_val, y_val)


def run_model():
    # np.seterr(all='raise')
    (x_train, y_train), _, _ = ln.datasets.mnist.load_data()
    m = 64
    x_train, y_train = x_train[:m, :], y_train[:m]
    y_train = ln.nn.one_hot(y_train, 10)
    print("run_model, data's shape: ", x_train.shape, y_train.shape)

    y = ln.placeholder("y")
    x = ln.placeholder("x")

    w1 = ln.get_variable(shape=(x_train.shape[1], 10), initializer=ln.initializers.normal, name="w1")
    b1 = ln.get_variable(shape=(1, 10), initializer=ln.initializers.zeros, name="b1")
    _z1 = ln.matmul(x, w1)
    z1 = ln.add(_z1, ln.broadcast(b1, _z1))
    a1 = ln.relu(z1)

    w2 = ln.get_variable(shape=(10, 10), initializer=ln.initializers.normal, name="w2")
    b2 = ln.get_variable(shape=(1, 10), initializer=ln.initializers.zeros, name="b2")
    _z2 = ln.matmul(a1, w2)
    z2 = ln.add(_z2, ln.broadcast(b2, _z2))
    y_hat = ln.nn.softmax(z2)

    # begin of test
    # w1 = ln.get_variable(shape=(128, x_train.shape[0]), initializer=ln.initializers.normal, name="w1")
    # z1 = ln.matmul(w1, x)
    #
    # w2 = ln.get_variable(shape=(10, 128), initializer=ln.initializers.normal, name="w2")
    # z2 = ln.matmul(w2, z1)
    # y_hat = ln.nn.softmax(z2)
    # end of test

    cost = ln.nn.cross_entropy(y_hat, y)
    opt = ln.optimizers.GradientDescent(0.01)
    optimizer = opt.minimize(cost)
    opt.gradient_check({x: x_train, y: y_train})
    for i in range(20001):
        optimizer.eval(feed_dict={x: x_train, y: y_train})
        if (i % 100) == 0:
            print("loss: {}, accuracy: {}.".format(cost.eval(), accuracy(y_hat.eval(), y_train)))


def main():
    run_tests()
    # test_model()
    # run_model()


if __name__ == "__main__":
    main()
