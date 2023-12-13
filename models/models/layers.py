import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

class AdaptiveInstanceNormalization(krs.layers.Layer):
    def __init__(self, epsilon=1e-5, momentum=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.mean = None
        self.std = None
    
    def build(self, input_shape):
        # Deconstruct input shape
        n_batch, n_time, n_mels, n_chans = input_shape
        self.n_chans = n_chans
        shape = (1, 1, 1, n_chans)

        # Build 
        self.mean = tf.Variable(tf.zeros(shape))
        self.std = tf.Variable(tf.ones(shape))

        # Run superclass build
        super().build(input_shape)
    
    def call(self, x, training=None):
        assert self.bias is not None and self.weight is not None

        if training != False:
            # Update moving average for mean & variance
            mu = tf.reshape(tf.math.reduce_mean(x, axis=(0, 1, 2)), (-1, 1, 1, self.n_chans))
            sigma = tf.reshape(tf.math.reduce_std(x, axis=(0, 1, 2)), (-1, 1, 1, self.n_chans))

            self.mean.assign((tf.constant(1.0) - self.momentum) * self.mean + self.momentum * mu)
            self.std.assign((tf.constant(1.0) - self.momentum) * self.std + self.momentum * sigma)
        
        # Normalize
        x_norm = (x - self.mean) / (self.std + self.epsilon)

        # Rescale
        weight = tf.reshape(self.weight, (-1, 1, 1, self.n_chans))
        bias = tf.reshape(self.bias, (-1, 1, 1, self.n_chans))
        x_scaled = x_norm * weight + bias
        
        return x_scaled
    
    def get_config(self):
        return {
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "weight": self.weight,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
        }

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
                        activation=self.activation,
                    )
                )
            else:
                self.conv_layers.append(
                    krs.layers.Conv2D(
                        self.filters,
                        self.kernel_size,
                        padding="same",
                        activation=self.activation,
                    )
                )
        super().build(input_shape)
    
    def call(self, inputs):
        x = inputs
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i == 0 and self.skip_connection:
                skip = x
        
        if self.skip_connection and len(self.conv_layers) > 1:
            # Everything past the first conv layer can be skipped
            x = tf.add(x, skip)
        elif self.skip_connection:
            print("WARNING: Skip connection has no effect if conv_depth = 1")
        
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
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.skip_connection = skip_connection
        self.activation = activation
        self.add_padding = add_padding
        self.pooling_type = pooling_type

    def build(self, input_shape):
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
        if self.pooling_type == "average":
            self.pooling_layer = krs.layers.AveragePooling2D(pool_size=(2, 2))
        elif self.pooling_type == "max":
            self.pooling_layer = krs.layers.MaxPooling2D(pool_size=(2, 2))
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
    ):
        super().__init__()
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

#@krs.saving.register_keras_serializable()
class Conv2DEncoder(krs.models.Model):
    def __init__(
        self, 
        layer_filters, 
        layer_padding,
        kernel_size,
        conv_depth,
        skip_connection=True,
        activation="relu",
        output_activation="relu",
        pooling_type="average",
    ):
        super().__init__()

        # Save parameters
        self.layer_filters = layer_filters
        self.layer_padding = layer_padding
        self.kernel_size = kernel_size
        self.conv_depth = conv_depth
        self.skip_connection = skip_connection
        self.activation = activation
        self.output_activation = output_activation
        self.pooling_type = pooling_type

        # Build encoder
        self.encoder = krs.models.Sequential()
        n_layers = len(self.layer_filters)
        for layer, (filters, padding) in enumerate(zip(self.layer_filters, self.layer_padding)):
            # No pooling for last layer
            #pooling_type = self.pooling_type if layer < n_layers - 1 else "none" 
            self.encoder.add(
                Conv2DEncoderBlock(
                    filters,
                    self.kernel_size,
                    self.conv_depth,
                    skip_connection=self.skip_connection,
                    activation=self.activation,
                    add_padding=padding,
                    pooling_type=self.pooling_type,
                )
            )
        # Add one more layer without pooling
        self.encoder.add(
            Conv2DEncoderBlock(
                self.layer_filters[-1],
                self.kernel_size,
                self.conv_depth,
                skip_connection=self.skip_connection,
                activation=self.output_activation,
                add_padding=False,
                pooling_type="none"
            )
        )

    def call(self, x):
        h = self.encoder(x)
        return h

#@krs.saving.register_keras_serializable()
class Conv2DDecoder(krs.models.Model):
    def __init__(
        self,
        layer_filters,
        layer_padding,
        kernel_size,
        conv_depth,
        skip_connection=True,
        activation="relu",
    ):
        super().__init__()

        # Save parameters
        self.layer_filters = layer_filters
        self.layer_padding = layer_padding
        self.kernel_size = kernel_size
        self.conv_depth = conv_depth
        self.skip_connection = skip_connection
        self.activation = activation

        # Build decoder
        self.decoder = krs.models.Sequential()
        n_layers = len(self.layer_filters)
        for layer, (filters, padding) in enumerate(zip(self.layer_filters, self.layer_padding)):
            self.decoder.add(
                Conv2DDecoderBlock(
                    filters,
                    self.kernel_size,
                    self.conv_depth,
                    skip_connection=self.skip_connection,
                    activation=self.activation,
                    remove_padding=padding,
                    upsampling=True,
                )
            )

    def call(self, h):
        x_hat = self.decoder(h)
        return x_hat

#@krs.saving.register_keras_serializable()
class MLP(krs.models.Model):
    def __init__(
        self, 
        hidden_dims, 
        output_dim,
        hidden_activation="relu",
        output_activation="relu",
    ):
        super().__init__()

        # Store params
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Start MLP with flattening layer
        self.mlp = krs.models.Sequential(
            krs.layers.Flatten()
        )

        # Add dense layers
        self.layer_dims = hidden_dims
        for units in self.layer_dims:
            self.mlp.add(
                krs.layers.Dense(
                    units,
                    activation=hidden_activation,
                )
            )
        self.mlp.add(
            krs.layers.Dense(
                output_dim
            )
        )
    
    def call(self, x):
        y_hat = self.mlp(x)
        return y_hat
    

class GAN(krs.models.Model):
    def __init__(self, generator, discriminator):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
    
    @property
    def metrics(self):
        return self.generator.metrics + self.discriminator.metrics

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.generator.compile(g_optimizer)
        self.discriminator.compile(d_optimizer)
    
    def train_step(self, data):
        # Train discriminator
        self.discriminator.train_step(data, self.generator)

        # Train generator
        self.generator.train_step(data, self.discriminator)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Test discriminator
        self.discriminator.test_step(data, self.generator)

        # Test generator
        self.generator.test_step(data, self.discriminator)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}