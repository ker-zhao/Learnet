# Learnet
**An Open Source Machine Learning Framework**
* Build a model to fit mnist dataset with Learnet only need to write few lines of code, like the example down below.
* GPU is supported in Learnet.  (Use: `ln.lib.enable_gpu(True)`)



## Example
```
import Learnet.learnet as ln

(x_train, y_train), (x_val, y_val), _ = ln.datasets.mnist.load_data()
model = ln.models.Sequential([
    ln.layers.Dense(128, input_dims=x_train.shape[1], activation="relu"),
    ln.layers.Dropout(0.2),
    ln.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="cross_entropy")
model.fit(x_train, y_train, epochs=5, verbose=1)
model.evaluate(x_val, y_val)
```

## Dependencies
* NumPy
* CUDA/CuPy (If you want to enable GPU.)
