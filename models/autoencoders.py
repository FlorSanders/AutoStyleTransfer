import tensorflow as tf
import tensorflow.keras as krs
import numpy as np
from .layers import Conv2DEncoder, Conv2DDecoder, MLP, AdaptiveInstanceNormalization

#@krs.saving.register_keras_serializable()
class Conv2DAutoencoder(krs.models.Model):
    def __init__(self, **params):
        super().__init__()

        # Save parameters to class
        self.params = params
        self.feature_shape = params.get("feature_shape")
        self.input_time, self.input_mels, self.input_chans = self.feature_shape
        self.compression = params.get("compression")
        self.kernel_size = params.get("kernel_size")
        self.conv_depth = params.get("conv_depth")
        self.input_chans_multiplier = params.get("input_chans_multiplier", 1)
        self.skip_connection = params.get("skip_connection", True)
        self.activation = params.get("activation", "relu")
        self.pooling_type = params.get("pooling_type", "average")

        # Determine nr of layers to be used (and check config validity)
        self.k = np.log2(self.input_chans_multiplier)
        assert np.round(self.k) == self.k, "input_chans_multiplier should be a power of 2"
        self.l = np.log2(self.compression)
        assert np.round(self.l) == self.l, "compression should be a power of 2"
        assert self.input_mels / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input mels dimension"
        assert self.input_time / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input time dimension"
        self.n_layers = int(np.round(self.k + self.l))

        # Determine nr of channels & padding for the encoder layers
        layer_filters = [self.input_chans] + [self.input_chans * self.input_chans_multiplier * 2**(n + 1) for n in range(self.n_layers)]
        layer_padding, time = [None] * self.n_layers, self.input_time
        for n in range(self.n_layers):
            layer_padding[n] = time % 2 == 1
            time = (time + layer_padding[n]) // 2

        # Build encoder
        self.encoder = Conv2DEncoder(
            layer_filters[1:],
            layer_padding,
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            pooling_type=self.pooling_type,
        )

        # Build decoder
        self.decoder = Conv2DDecoder(
            layer_filters[::-1][1:],
            layer_padding[::-1],
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
        )
    
    def call(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat

#@krs.saving.register_keras_serializable()
class VariationalAutoencoder(krs.models.Model):
    # Based on: https://keras.io/examples/generative/vae/

    def __init__(self, **params):
        super().__init__()

        # Save parameters to class
        self.params = params
        self.feature_shape = params.get("feature_shape")
        self.input_time, self.input_mels, self.input_chans = self.feature_shape
        self.compression = params.get("compression")
        self.kernel_size = params.get("kernel_size")
        self.conv_depth = params.get("conv_depth")
        self.input_chans_multiplier = params.get("input_chans_multiplier", 1)
        self.skip_connection = params.get("skip_connection", True)
        self.activation = params.get("activation", "relu")
        self.pooling_type = params.get("pooling_type", "average")
        self.kl_reg = params.get("kl_reg")
        self.vol_reg = params.get("vol_reg")

        # Determine nr of layers to be used (and check config validity)
        self.k = np.log2(self.input_chans_multiplier)
        assert np.round(self.k) == self.k, "input_chans_multiplier should be a power of 2"
        self.l = np.log2(self.compression)
        assert np.round(self.l) == self.l, "compression should be a power of 2"
        assert self.input_mels / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input mels dimension"
        assert self.input_time / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input time dimension"
        self.n_layers = int(np.round(self.k + self.l))

        # Determine nr of channels & padding for the encoder layers
        layer_filters = [self.input_chans] + [self.input_chans * self.input_chans_multiplier * 2**(n + 1) for n in range(self.n_layers)]
        layer_padding, time = [None] * self.n_layers, self.input_time
        for n in range(self.n_layers):
            layer_padding[n] = time % 2 == 1
            time = (time + layer_padding[n]) // 2

        # Build encoders
        self.mu_encoder = Conv2DEncoder(
            layer_filters[1:],
            layer_padding,
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            pooling_type=self.pooling_type,
        )

        self.sigma2_encoder = Conv2DEncoder(
            layer_filters[1:],
            layer_padding,
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            pooling_type=self.pooling_type
        )

        # Build decoder
        self.decoder = Conv2DDecoder(
            layer_filters[::-1][1:],
            layer_padding[::-1],
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
        )

        # Keep track of losses
        self.loss_tracker = krs.metrics.Mean(name="loss") # total loss
        self.r_loss_tracker = krs.metrics.Mean(name="r_loss") # reconstruction loss
        self.kl_loss_tracker = krs.metrics.Mean(name="kl_loss") # kullback leibler loss
        self.vol_loss_tracker = krs.metrics.Mean(name="vol_loss") # Volume loss
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.r_loss_tracker,
            self.kl_loss_tracker,
            self.vol_loss_tracker,
        ]

    def encode(self, x):
        mu = self.mu_encoder(x)
        logsigma2 = self.sigma2_encoder(x)
        return mu, logsigma2
    
    def reparametrize(self, mu, logsigma2):
        z = tf.random.normal(shape=logsigma2.shape[1:])
        h = mu + tf.exp(tf.clip_by_value(logsigma2, 0., 10.)) * z
        return h
    
    def decode(self, h):
        x_hat = self.decoder(h)
        return x_hat

    def call(self, x):
        mu, logsigma2 = self.encode(x)
        h = self.reparametrize(mu, logsigma2)
        x_hat = self.decode(h)
        return x_hat
    
    def compute_r_loss(self, x, x_hat):
        r_loss = tf.reduce_mean(tf.abs(x - x_hat)) # MeanAbsoluteError
        self.r_loss_tracker.update_state(r_loss)
        return r_loss

    def compute_kl_loss(self, mu, logsigma2):
        # See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        kl_loss = tf.reduce_mean(1/2 * (tf.exp(tf.clip_by_value(logsigma2, 0., 10.)) + mu**2 - logsigma2 - 1))
        self.kl_loss_tracker.update_state(kl_loss)
        return kl_loss
    
    def compute_vol_loss(self, x, x_hat):
        in_vol = tf.reduce_mean(tf.abs(x), axis=(1,2,3))
        out_vol = tf.reduce_mean(tf.abs(x_hat), axis=(1,2,3))
        vol_loss = tf.reduce_mean(tf.abs(out_vol - in_vol))
        self.vol_loss_tracker.update_state(vol_loss)
        return vol_loss

    def compute_loss(self, x, x_hat, mu, logsigma2):
        r_loss = self.compute_r_loss(x, x_hat)
        kl_loss = self.compute_kl_loss(x, x_hat)
        vol_loss = self.compute_vol_loss(x, x_hat)
        loss = r_loss + self.kl_reg * kl_loss + self.vol_reg * vol_loss
        self.loss_tracker.update_state(loss)
        return loss
    
    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward propagation
        with tf.GradientTape() as tape:
            # Autoencode
            mu, logsigma2 = self.encode(x)
            h = self.reparametrize(mu, logsigma2)
            x_hat = self.decode(h)

            # Compute loss factors
            loss = self.compute_loss(x, x_hat, mu, logsigma2)
        
        # Backpropagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, self.trainable_weights)

        # Optimization step
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack data
        x, y = data

        # Autoencode
        mu, logsigma2 = self.encode(x)
        h = self.reparametrize(mu, logsigma2)
        x_hat = self.decode(h)

        # Commute loss factors
        loss = self.compute_loss(x, x_hat, mu, logsigma2)

        # Return losses
        return {m.name: m.result() for m in self.metrics}

#@krs.saving.register_keras_serializable()
class GANDiscriminator(krs.models.Model):
    def __init__(self, **params):
        super().__init__()
        
        # Store parameters
        self.params = params
        self.feature_shape = params.get("feature_shape")
        self.input_time, self.input_mels, self.input_chans = self.feature_shape
        self.feature_size = np.prod(self.feature_shape)
        self.mlp_layers = params.get("mlp_layers")
        self.conv_layers = params.get("conv_layers")
        self.conv_kernel_size = params.get("conv_kernel_size")
        self.conv_pooling_size = params.get("conv_pooling_size")
        self.conv_pooling_type = params.get("conv_pooling_type", "max")

        # Store constants
        self.conv_depth = 1
        self.conv_input_chans_multiplier = 1
        self.conv_skip_connection = False
        self.conv_activation = "relu"
        self.mlp_activation = "relu"
        self.output_activation = "sigmoid"
        self.output_size = 1
        
        # Figure out layer sizes
        conv_channels = np.logspace(1, self.conv_layers, self.conv_layers, base=2).astype(int)
        print(f"{conv_channels = }")
        conv_output_size = round(self.feature_size / (self.conv_pooling_size**(2 * self.conv_layers)) * conv_channels[-1])
        print(f"{conv_output_size = }")
        mlp_sizes = np.round(np.logspace(np.log2(1), np.log2(conv_output_size), self.mlp_layers + 1, base=2)[:-1][::-1]).astype(int)
        print(f"{mlp_sizes = }")
           
        # CNN Feature extractor
        self.cnn = krs.models.Sequential()
        for conv_layer in range(self.conv_layers):
            self.cnn.add(
                krs.layers.Conv2D(
                    conv_channels[conv_layer],
                    self.conv_kernel_size,
                    padding="same",
                    activation=self.conv_activation,
                )
            )
            if self.conv_pooling_type == "average":
                self.cnn.add(krs.layers.AveragePooling2D(pool_size=self.conv_pooling_size))
            elif self.conv_pooling_type == "max":
                self.cnn.add(krs.layers.MaxPooling2D(pool_size=self.conv_pooling_size))
            else:
                raise ValueError(f"Unknown pooling type: {pooling_type}")

        # MLP classifier
        self.mlp = MLP(
            mlp_sizes[:-1],
            mlp_sizes[-1],
            hidden_activation=self.mlp_activation,
            output_activation=self.output_activation,
        )

        # Keep track of losses
        self.d_loss_tracker = krs.metrics.Mean(name="d_loss")

    @property
    def metrics(self):
        return [self.d_loss_tracker]

    def compute_loss(self, p_real, p_fake):
        # RaGAN loss
        # The discriminator wants p_real to be close to 1 and p_fake to be close to 0
        real_loss = tf.reduce_mean(((p_real - 1) - tf.reduce_mean(p_fake))**2)
        fake_loss = tf.reduce_mean((p_fake - (tf.reduce_mean(p_real) - 1))**2)
        d_loss = (real_loss + fake_loss) / 2
        self.d_loss_tracker.update_state(d_loss)
        return d_loss
    
    def call(self, x):
        # Extract features
        h = self.cnn(x)
        # Classify
        p = self.mlp(h)
        
        return p

    def train_step(self, data, generator):
        # Unpack data
        x, y = data

        # Generate real & fake sample
        x_hat, x_fake = generator(x)

        # Forward propagation
        with tf.GradientTape() as tape:
            # Discriminate
            p_real = self.call(x)
            p_fake = self.call(x_fake)

            # Compute loss
            loss = self.compute_loss(p_real, p_fake)

        # Backward propagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        # Optimizer step
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data, generator):
        # Unpack data
        x, y = data

        # Generate real & fake sample
        x_hat, x_fake = generator(x)

        # Discriminate
        p_real = self.call(x)
        p_fake = self.call(x_fake)

        # Compute loss
        loss = self.compute_loss(p_real, p_fake)

        # Return loss
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        return self.params

#@krs.saving.register_keras_serializable()
class GANGenerator(krs.models.Model):
    def __init__(self, **params):
        super().__init__()

        # Save parameters to class
        self.params = params
        self.feature_shape = params.get("feature_shape")
        self.input_time, self.input_mels, self.input_chans = self.feature_shape
        self.compression = params.get("compression")
        self.kernel_size = params.get("kernel_size")
        self.conv_depth = params.get("conv_depth")
        self.input_chans_multiplier = params.get("input_chans_multiplier", 1)
        self.skip_connection = params.get("skip_connection", True)
        self.activation = params.get("activation", "relu")
        self.hidden_activation = params.get("hidden_activation", "sigmoid")
        self.pooling_type = params.get("pooling_type", "average")
        self.gan_reg = params.get("gan_reg")
        self.c_reg = params.get("c_reg")
        self.s_reg = params.get("s_reg")
        self.mode = params.get("mode")
        assert self.mode in ["concatenate", "add", "adain"]

        # Determine nr of layers to be used (and check config validity)
        k = np.log2(self.input_chans_multiplier)
        assert np.round(k) == k, "input_chans_multiplier should be a power of 2"
        l = np.log2(self.compression)
        assert np.round(l) == l, "compression should be a power of 2"
        assert self.input_mels / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input mels dimension"
        assert self.input_time / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input time dimension"
        self.n_layers = int(np.round(k + l))

        # Determine nr of channels & padding for the encoder layers
        layer_filters = [self.input_chans] + [self.input_chans * self.input_chans_multiplier * 2**(n + 1) for n in range(self.n_layers)]
        layer_padding, time = [None] * self.n_layers, self.input_time
        for n in range(self.n_layers):
            layer_padding[n] = time % 2 == 1
            time = (time + layer_padding[n]) // 2
        
        # Build encoders
        self.content_encoder = Conv2DEncoder(
            layer_filters[1:],
            layer_padding,
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            output_activation=self.activation,
            pooling_type=self.pooling_type
        )
        
        if self.mode != "concatenate":
            # Use one encoder for both style & content
            self.style_encoder = Conv2DEncoder(
                layer_filters[1:],
                layer_padding,
                self.kernel_size,
                self.conv_depth,
                skip_connection=self.skip_connection,
                activation=self.activation,
                output_activation=self.hidden_activation,
                pooling_type=self.pooling_type,
            )

        # Build decoder
        self.decoder = Conv2DDecoder(
            layer_filters[::-1][1:],
            layer_padding[::-1],
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
        )

        # Keep track of losses
        self.g_loss_tracker = krs.metrics.Mean(name="loss") # Total loss
        self.r_loss_tracker = krs.metrics.Mean(name="r_loss") # Reconstruction loss
        self.gan_loss_tracker = krs.metrics.Mean(name="gan_loss") # Adverserial loss
        self.c_loss_tracker = krs.metrics.Mean(name="c_loss") # Content loss
        self.s_loss_tracker = krs.metrics.Mean(name="s_loss") # Style loss
    
    @property
    def metrics(self):
        return [self.g_loss_tracker, self.r_loss_tracker, self.gan_loss_tracker, self.c_loss_tracker, self.s_loss_tracker]
    
    def encode(self, x):
        if self.mode == "concatenate":
            h = self.content_encoder(x)
            content, style = tf.split(h, 2, axis=3)
        else:
            content = self.content_encoder(x)
            style = self.style_encoder(x)
        return content, style

    def sample_style(self, style):
        #print(f"{style.shape = }")
        #sigma = tf.reshape(tf.math.reduce_std(style, axis=(1,2,3)) / tf.math.sqrt(12.), (-1, 1, 1, 1))
        #print(f"{sigma.shape = }")
        #style_fake = tf.expand_dims(tf.random.uniform(shape = style.shape[1:]), axis=0) * sigma
        #print(f"{style_fake.shape = }")
        if self.mode == "adain":
            style_fake = tf.expand_dims(tf.random.normal(shape = style.shape[1:]), axis=0)
        else:
            style_fake = tf.expand_dims(tf.random.uniform(shape = style.shape[1:]), axis=0)
        return style_fake

    def decode(self, content, style):
        if self.mode == "concatenate":
            style = tf.ones_like(content) * style
            h = tf.concat((content, style), axis=3)
        elif self.mode == "add":
            h = (content + style) / 2
        elif self.mode == "adain":
            # Compute mean & std of content
            eps = krs.backend.epsilon()
            content_mean, content_variance = tf.nn.moments(content, axes=(1, 2), keepdims=True)
            content_std = tf.math.sqrt(content_variance + eps)
            # Compute mean & std of style
            style_mean, style_variance = tf.nn.moments(style, axes=(1, 2), keepdims=True)
            style_std = tf.math.sqrt(style_variance + eps)
            # Perform adain style transfer
            h = style_std * (content - content_mean) / content_std + style_mean
        else:
            raise AssertionError(f"Mode {self.mode} is not known")
        x_hat = self.decoder(h)
        return x_hat

    def call(self, x):
        # Encode sample to content & style
        content, style = self.encode(x)

        # Generate fake style
        style_fake = self.sample_style(style)

        # Decode real and fake sample
        x_hat = self.decode(content, style)
        x_fake = self.decode(content, style_fake)

        return x_hat, x_fake

    def compute_r_loss(self, x, x_hat):
        r_loss = tf.reduce_mean(tf.abs(x - x_hat)) # MeanAbsoluteError
        self.r_loss_tracker.update_state(r_loss)
        return r_loss
    
    def compute_gan_loss(self, p_real, p_fake):
        # RaGAN loss
        # The generator wants p_real to be close to 0 and p_fake to be close to 1
        real_loss = tf.reduce_mean((p_real - (tf.reduce_mean(p_fake) - 1))**2)
        fake_loss = tf.reduce_mean(((p_fake - 1) - tf.reduce_mean(p_real))**2)
        gan_loss = (real_loss + fake_loss) / 2
        self.gan_loss_tracker.update_state(gan_loss)
        return gan_loss

    def compute_c_loss(self, content, content_hat):
        c_loss = tf.reduce_mean(tf.abs(content - content_hat))
        self.c_loss_tracker.update_state(c_loss)
        return c_loss
    
    def compute_s_loss(self, style_fake, style_fake_hat):
        s_loss = tf.reduce_mean(tf.abs(style_fake - style_fake_hat))
        self.s_loss_tracker.update_state(s_loss)
        return s_loss
    
    def compute_loss(
        self,
        x, 
        content, 
        style, 
        style_fake, 
        x_hat, 
        x_fake, 
        p_real, 
        p_fake, 
        content_hat, 
        style_fake_hat
    ):
        # Compute loss contributions
        r_loss = self.compute_r_loss(x, x_hat)
        gan_loss = self.compute_gan_loss(p_real, p_fake)
        c_loss = self.compute_c_loss(content, content_hat)
        s_loss = self.compute_s_loss(style_fake, style_fake_hat)

        g_loss = r_loss + self.gan_reg * gan_loss + self.c_reg * c_loss + self.s_reg * s_loss
        self.g_loss_tracker.update_state(g_loss)
        return g_loss
    
    def train_step(self, data, discriminator):
        # Unpack data
        x, y = data

        # Forward propagation
        with tf.GradientTape() as tape:
            # Encode real
            content, style = self.encode(x)

            # Create fake style
            style_fake = self.sample_style(style)

            # Decode real and fake
            x_hat = self.decode(content, style)
            x_fake = self.decode(content, style_fake)

            # Discriminate
            p_real = discriminator(x)
            p_fake = discriminator(x_fake)
        
            # Encode fake
            content_hat, style_fake_hat = self.encode(x_fake)

            # Compute loss
            loss = self.compute_loss(
                x, 
                content, 
                style, 
                style_fake, 
                x_hat, 
                x_fake, 
                p_real, 
                p_fake, 
                content_hat, 
                style_fake_hat
            )
        
        # Backpropagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        # Optimization step
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data, discriminator):
        # Unpack data
        x, y = data

        # Encode real
        content, style = self.encode(x)

        # Create fake style
        style_fake = self.sample_style(style)

        # Decode real and fake
        x_hat = self.decode(content, style)
        x_fake = self.decode(content, style_fake)

        # Discriminate
        p_real = discriminator(x)
        p_fake = discriminator(x_fake)
    
        # Encode fake
        content_hat, style_fake_hat = self.encode(x_fake)

        # Compute loss
        loss = self.compute_loss(
            x, 
            content, 
            style, 
            style_fake, 
            x_hat, 
            x_fake, 
            p_real, 
            p_fake, 
            content_hat, 
            style_fake_hat
        )

        # Return losses
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        return self.params

#@krs.saving.register_keras_serializable()
class MUNITGenerator(krs.models.Model):
    def __init__(self, **params):
        super().__init__()
    
        self.params = params
        self.feature_shape =  params.get("feature_shape")
        self.input_time, self.input_mels, self.input_chans = self.feature_shape
        self.compression = params.get("compression")
        self.style_dim = params.get("style_dim")
        self.kernel_size = params.get("kernel_size")
        self.conv_depth = params.get("conv_depth")
        self.input_chans_multiplier = params.get("input_chans_multiplier", 1)
        self.skip_connection = params.get("skip_connection", True)
        self.activation = params.get("activation", "relu")
        self.pooling_type = params.get("pooling_type", "average")
        self.adain_momentum = params.get("adain_momentum")
        self.adain_epsilon = params.get("adain_epsilon")
        self.gan_reg = params.get("gan_reg")
        self.c_reg = params.get("c_reg")
        self.s_reg = params.get("s_reg")

        # Determine nr of layers to be used (and check config validity)
        k = np.log2(self.input_chans_multiplier)
        assert np.round(k) == k, "input_chans_multiplier should be a power of 2"
        l = np.log2(self.compression)
        assert np.round(l) == l, "compression should be a power of 2"
        assert self.input_mels / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input mels dimension"
        assert self.input_time / (self.compression * self.input_chans_multiplier) >= 1, "input_chans_multiplier * compression should be smaller than input time dimension"
        self.n_layers = int(np.round(k + l))
    
        # Determine nr of channels & padding for the encoder layers
        layer_filters = [self.input_chans] + [self.input_chans * self.input_chans_multiplier * 2**(n + 1) for n in range(self.n_layers)]
        layer_padding, time = [None] * self.n_layers, self.input_time
        for n in range(self.n_layers):
            layer_padding[n] = time % 2 == 1
            time = (time + layer_padding[n]) // 2

        # Build encoders
        self.content_encoder = Conv2DEncoder(
            layer_filters[1:],
            layer_padding,
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            output_activation=self.activation,
            pooling_type=self.pooling_type
        )
        self.style_encoder = Conv2DEncoder(
            layer_filters[1:],
            layer_padding,
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
            output_activation=self.activation,
            pooling_type=self.pooling_type,
        )
        self.style_reducer = krs.models.Sequential([
            krs.layers.Conv2D(
                self.style_dim,
                self.kernel_size,
                padding="valid",
                activation=self.activation,
            ),
            krs.layers.GlobalAveragePooling2D(),
        ])

        # Build decoder
        self.decoder = Conv2DDecoder(
            layer_filters[::-1][1:],
            layer_padding[::-1],
            self.kernel_size,
            self.conv_depth,
            skip_connection=self.skip_connection,
            activation=self.activation,
        )
        self.style_mlp = MLP(
            [layer_filters[-1]],
            layer_filters[-1] * 2,
            hidden_activation=self.activation,
            output_activation=self.activation,
        )
        self.style_adain = AdaptiveInstanceNormalization(
            self.adain_epsilon,
            self.adain_momentum,
        )

        # Keep track of losses
        self.g_loss_tracker = krs.metrics.Mean(name="loss") # Total loss
        self.r_loss_tracker = krs.metrics.Mean(name="r_loss") # Reconstruction loss
        self.gan_loss_tracker = krs.metrics.Mean(name="gan_loss") # Adverserial loss
        self.c_loss_tracker = krs.metrics.Mean(name="c_loss") # Content loss
        self.s_loss_tracker = krs.metrics.Mean(name="s_loss") # Style loss
    
    @property
    def metrics(self):
        return [self.g_loss_tracker, self.r_loss_tracker, self.gan_loss_tracker, self.c_loss_tracker, self.s_loss_tracker]
    
    def encode(self, x):
        # Encode content
        content = self.content_encoder(x)
        # Encode style & reduce to style_dim
        style_full = self.style_encoder(x)
        style = self.style_reducer(style_full)
        return content, style
    
    def sample_style(self, style):
        style_fake = tf.expand_dims(tf.random.normal(shape = style.shape[1:]), axis=0)
        return style_fake

    def decode(self, content, style):
        # Compute and assign adain_parameters
        adain_params = self.style_mlp(style)
        mean, var = tf.split(adain_params, 2, axis=1)
        self.style_adain.bias = mean
        self.style_adain.weight = var
        
        # Rescale content using adain
        h = self.style_adain(content)

        # Decode
        x_hat = self.decoder(h)

        return x_hat

    def call(self, x):
        # Encode sample to content & style
        content, style = self.encode(x)

        # Generate fake style
        style_fake = self.sample_style(style)

        # Decode real and fake sample
        x_hat = self.decode(content, style)
        x_fake = self.decode(content, style_fake)

        return x_hat, x_fake
    
    def compute_r_loss(self, x, x_hat):
        r_loss = tf.reduce_mean(tf.abs(x - x_hat)) # MeanAbsoluteError
        self.r_loss_tracker.update_state(r_loss)
        return r_loss
    
    def compute_gan_loss(self, p_real, p_fake):
        # RaGAN loss
        # The generator wants p_real to be close to 0 and p_fake to be close to 1
        real_loss = tf.reduce_mean((p_real - (tf.reduce_mean(p_fake) - 1))**2)
        fake_loss = tf.reduce_mean(((p_fake - 1) - tf.reduce_mean(p_real))**2)
        gan_loss = (real_loss + fake_loss) / 2
        self.gan_loss_tracker.update_state(gan_loss)
        return gan_loss

    def compute_c_loss(self, content, content_hat):
        c_loss = tf.reduce_mean(tf.abs(content - content_hat))
        self.c_loss_tracker.update_state(c_loss)
        return c_loss
    
    def compute_s_loss(self, style_fake, style_fake_hat):
        s_loss = tf.reduce_mean(tf.abs(style_fake - style_fake_hat))
        self.s_loss_tracker.update_state(s_loss)
        return s_loss
    
    def compute_loss(
        self,
        x, 
        content, 
        style, 
        style_fake, 
        x_hat, 
        x_fake, 
        p_real, 
        p_fake, 
        content_hat, 
        style_fake_hat
    ):
        # Compute loss contributions
        r_loss = self.compute_r_loss(x, x_hat)
        gan_loss = self.compute_gan_loss(p_real, p_fake)
        c_loss = self.compute_c_loss(content, content_hat)
        s_loss = self.compute_s_loss(style_fake, style_fake_hat)

        g_loss = r_loss + self.gan_reg * gan_loss + self.c_reg * c_loss + self.s_reg * s_loss
        self.g_loss_tracker.update_state(g_loss)
        return g_loss
    
    def train_step(self, data, discriminator):
        # Unpack data
        x, y = data

        # Forward propagation
        with tf.GradientTape() as tape:
            # Encode real
            content, style = self.encode(x)

            # Create fake style
            style_fake = self.sample_style(style)

            # Decode real and fake
            x_hat = self.decode(content, style)
            x_fake = self.decode(content, style_fake)

            # Discriminate
            p_real = discriminator(x)
            p_fake = discriminator(x_fake)
        
            # Encode fake
            content_hat, style_fake_hat = self.encode(x_fake)

            # Compute loss
            loss = self.compute_loss(
                x, 
                content, 
                style, 
                style_fake, 
                x_hat, 
                x_fake, 
                p_real, 
                p_fake, 
                content_hat, 
                style_fake_hat
            )
        
        # Backpropagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        # Optimization step
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data, discriminator):
        # Unpack data
        x, y = data

        # Encode real
        content, style = self.encode(x)

        # Create fake style
        style_fake = self.sample_style(style)

        # Decode real and fake
        x_hat = self.decode(content, style)
        x_fake = self.decode(content, style_fake)

        # Discriminate
        p_real = discriminator(x)
        p_fake = discriminator(x_fake)
    
        # Encode fake
        content_hat, style_fake_hat = self.encode(x_fake)

        # Compute loss
        loss = self.compute_loss(
            x, 
            content, 
            style, 
            style_fake, 
            x_hat, 
            x_fake, 
            p_real, 
            p_fake, 
            content_hat, 
            style_fake_hat
        )

        # Return losses
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        return self.params
