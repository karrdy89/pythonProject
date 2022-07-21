import tensorflow as tf
import numpy as np

import os
ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)


model=tf.keras.Sequential([
    tf.keras.layers.Dense(1,input_shape=[1])
])
model.compile(optimizer="sgd", loss="mse")
model.fit(xs, ys, epochs=1000)

model.save(ROOT_DIR+'/models/test/1')