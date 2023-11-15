import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

class Conv2DBlock(krs.layers.Layer):
    """
    Block with a sequence of Conv2D layers (with the same parameters) and an optional skip connection
    More info on custom layers: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    """

    def __init__(
        self, 
        filters, 
        kernel_size, 
        depth, 
        skip_connection=True, 
        activation="relu", 
        is_transpose=False, 
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.skip_connection = skip_connection
        self.activation = activation
        self.is_transpose = is_transpose
    
    def build(self, input_shape):
        self.conv_layers = []
        for _ in range(self.depth):
            if self.is_transpose:
                self.conv_layers.append(
                    krs.layers.Conv2DTranspose(
                        self.filters,
                        self.kernel_size,
                        padding="same",
                        activation="relu",
                    )
                )
            else:
                self.conv_layers.append(
                    krs.layers.Conv2D(
                        self.filters,
                        self.kernel_size,
                        padding="same",
                        activation=self.activation
                    )
                )
        super().build(input_shape)
    
    def call(self, inputs):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        if self.skip_connection:
            x += inputs
        
        return x

    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "depth": self.depth,
            "skip_connection": self.skip_connection,
            "activation": self.activation,
            "is_transpose": self.is_transpose,
        }


class Conv2DEncoderBlock(krs.layers.Layer):
    """
    Encoder building block that halves the feature dimensions into a channel latent space using a sequence of identical Conv2D layers.
    Padding can be used to ensure all data is kept along the time dimension.
    More info on custom layers: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    """

    def __init__(
        self, 
        filters, 
        kernel_size, 
        depth, 
        skip_connection=True, 
        activation="relu", 
        add_padding=True, 
        pooling_type="average", 
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.skip_connection = skip_connection
        self.activation = activation
        self.add_padding = add_padding
        self.pooling_type = pooling_type

    def build(self, input_shape):
        print(f"{input_shape = }")
        # Conv2D Block
        self.conv_block = Conv2DBlock(
            self.filters,
            self.kernel_size,
            self.depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            is_transpose=False
        )

        # Padding layer (if time dimension is odd)
        padding = 1 if self.add_padding else 0
        self.padding_layer = krs.layers.ZeroPadding2D(
            padding=((0, padding), (0, 0))
        )

        # Add pooling layer
        pool_size = 2
        if self.pooling_type == "average":
            self.pooling_layer = krs.layers.AveragePooling2D(pool_size=(pool_size, pool_size), pool_stride=(pool_size, pool_size))
        elif self.pooling_type == "max":
            self.pooling_layer = krs.layers.MaxPooling2D(pool_size=(pool_size, pool_size), pool_stride=(pool_size, pool_size))
        elif self.pooling_type == "none":
            self.pooling_layer = None
        else:
            raise AssertionError(f"Unknown pooling type {self.pooling_type}")
    
        super().build(input_shape)
    
    def call(self, inputs):
        x = inputs
        x = self.conv_block(x)
        x = self.padding_layer(x)
        if self.pooling_layer is not None:
            x = self.pooling_layer(x)
        return x
    
    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "depth": self.depth,
            "skip_connection": self.skip_connection,
            "activation": self.activation,
            "add_padding": self.add_padding,
            "pooling_type": self.pooling_type,
        }

class Conv2DDecoderBlock(krs.layers.Layer):
    def __init__(
        self, 
        filters, 
        kernel_size, 
        depth, 
        skip_connection = True, 
        activation = "relu", 
        remove_padding=True, 
        upsampling=True, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.skip_connection = skip_connection
        self.activation = activation
        self.remove_padding = remove_padding
        self.upsampling = upsampling
    
    def build(self, input_shape):
        # Upsampling block
        if self.upsampling:
            self.upsampling_layer = krs.layers.UpSampling2D(size=(2,2))
        else:
            self.upsampling_layer = None

        # Cropping block
        padding = 1 if self.remove_padding else 0
        self.cropping_layer = krs.layers.Cropping2D(
            cropping=((0, padding), (0, 0))
        )

        # Convolution layer
        self.conv_block = Conv2DBlock(
            self.filters,
            self.kernel_size,
            self.depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            is_transpose=True
        )

        super().build(input_shape)
    
    def call(self, inputs):
        x = inputs
        if self.upsampling_layer is not None:
            x = self.upsampling_layer(x)
        x = self.cropping_layer(x)
        x = self.conv_block(x)
        return x

    def get_config(self):
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "depth": self.depth,
            "skip_connection": self.skip_connection,
            "activation": self.activation,
            "remove_padding": self.remove_padding,
            "upsampling": self.upsampling,
        }
