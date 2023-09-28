import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

class TransformLayer(krs.layers.Layer):
    def __init__(self, transform_function, **kwargs):
        super().__init__()
        self.transform_function = transform_function
        self.kwargs = kwargs
    
    def build(self, input_shape):
        self.transform_shape = self.transform_function(np.zeros(input_shape, dtype=float)).shape

    def call(self, x):
        if tf.is_symbolic_tensor(x):
            print("SYMBOLIC TENSOR")
            return tf.zeros(self.transform_shape)
        x_array = np.array(x).astype(float)
        x_transform = self.transform_function(x_array .astype(float), **self.kwargs)
        return tf.convert_to_tensor(x_transform, dtype=tf.float32)