import tensorflow as tf
import tensorflow.keras as krs
import numpy as np
from .autoencoders import Conv2DAutoencoder, VariationalAutoencoder, GANGenerator, GANDiscriminator, MUNITGenerator

@krs.saving.register_keras_serializable()
class Conv2DTranscoder(krs.models.Model):
    def __init__(self, **params):
        super().__init__()

        # Pop transcoder params
        self.h_reg = params.pop("h_reg")
        self.kl_reg = params.pop("kl_reg")

        # Initialize two identical autoencoders
        self.coderX = Conv2DAutoencoder(**params)
        self.coderY = Conv2DAutoencoder(**params)

        # Keep track of losses
        self.loss_tracker = krs.metrics.Mean(name="loss") # total loss
        self.r_loss_tracker = krs.metrics.Mean(name="r_loss") # Reconstruction loss
        self.h_loss_tracker = krs.metrics.Mean(name="h_loss") # Cross-encoding reconstruction loss
        self.kl_loss_tracker = krs.metrics.Mean(name="kl_loss") # Transcoding kullback-leibler loss
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.r_loss_tracker,
            self.h_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(self, I, Xto=True):
        if Xto:
            h = self.coderX.encoder(I)
        else:
            h = self.coderY.encoder(I)
        return h
    
    def decode(self, h, toY=True):
        if toY:
            O = self.coderY.decoder(h)
        else:
            O = self.coderX.decoder(h)
        return O          
    
    def transcode(self, I, XtoY=True):
        h = self.encode(I, Xto=XtoY)
        O = self.decode(h, toY=XtoY)
        return O
    
    def call(self, data):
        # Unpack data
        X, Y = data
        # Transcode in both directions
        Y_fake = self.transcode(X, XtoY=True)
        X_fake = self.transcode(Y, XtoY=False)
        return Y_fake, X_fake

    def compute_r_loss(self, X, X_hat, Y, Y_hat):
        rX_loss = tf.reduce_mean(tf.abs(X - X_hat))
        rY_loss = tf.reduce_mean(tf.abs(Y - Y_hat))
        r_loss = (rX_loss + rY_loss) / 2
        self.r_loss_tracker.update_state(r_loss)
        return r_loss
    
    def compute_h_loss(self, hX, hX_hat, hY, hY_hat):
        hX_loss = tf.reduce_mean(tf.abs(hX - hX_hat))
        hY_loss = tf.reduce_mean(tf.abs(hY - hY_hat))
        h_loss = (hX_loss + hY_loss) / 2
        self.h_loss_tracker.update_state(h_loss)
        return h_loss
    
    def compute_kl_loss(self, hX, hY):
        # Kullback Leibler Loss (hX = logsigma2) --> dissimilarity from N(0,1)
        klX_loss = tf.reduce_mean(1/2 * (tf.exp(tf.clip_by_value(hX, 0., 10.)) - hX - 1))
        klY_loss = tf.reduce_mean(1/2 * (tf.exp(tf.clip_by_value(hY, 0., 10.)) - hY - 1))
        # Kullback Leibler Loss --> dissimilarity between one another
        kl_loss = tf.math.abs(klX_loss - klY_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return kl_loss

    def compute_loss(self, X, Y, hX, hY, X_hat, Y_hat, X_fake, Y_fake, hX_hat, hY_hat):
        # Compute loss contributions
        r_loss = self.compute_r_loss(X, X_hat, Y, Y_hat)
        h_loss = self.compute_h_loss(hX, hY, hX_hat, hY_hat)
        kl_loss = self.compute_kl_loss(hX, hY)

        # Compute total loss
        loss = r_loss + self.h_reg * h_loss + self.kl_reg * kl_loss
        self.loss_tracker.update_state(loss)
        return loss

    
    def train_step(self, data):
        # Unpack data
        X, Y = data

        # Forward propagation
        with tf.GradientTape() as tape:
            # Input Encoding
            hX = self.encode(X, Xto=True)
            hY = self.encode(Y, Xto=False)

            # Self Decoding
            X_hat = self.decode(hX, toY=False)
            Y_hat = self.decode(hY, toY=True)

            # Cross Decoding
            Y_fake = self.decode(hX, toY=True)
            X_fake = self.decode(hY, toY=False)

            # Cross Encoding
            hX_hat = self.encode(Y_fake, Xto=False)
            hY_hat = self.encode(X_fake, Xto=True)

            # Compute loss
            loss = self.compute_loss(X, Y, hX, hY, X_hat, Y_hat, X_fake, Y_fake, hX_hat, hY_hat)

        # Backpropagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, self.trainable_weights)
        
        # Optimization step
        self.optimizer.apply_gradients(zip(gradients,  trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack data
        X, Y = data

        # Input Encoding
        hX = self.encode(X, Xto=True)
        hY = self.encode(Y, Xto=False)

        # Self Decoding
        X_hat = self.decode(hX, toY=False)
        Y_hat = self.decode(hY, toY=True)

        # Cross Decoding
        Y_fake = self.decode(hX, toY=True)
        X_fake = self.decode(hY, toY=False)

        # Cross Encoding
        hX_hat = self.encode(Y_fake, Xto=False)
        hY_hat = self.encode(X_fake, Xto=True)

        # Compute loss
        loss = self.compute_loss(X, Y, hX, hY, X_hat, Y_hat, X_fake, Y_fake, hX_hat, hY_hat)

        # Return losses
        return {m.name: m.result() for m in self.metrics}

@krs.saving.register_keras_serializable()
class VariationalTranscoder(krs.models.Model):
    def __init__(self, **params):
        super().__init__()

        # Pop transcoder params
        self.h_reg = params.pop("h_reg")
        self.kl_reg = params.get("kl_reg")

        # Initialize two identical autoencoders
        self.coderX = VariationalAutoencoder(**params)
        self.coderY = VariationalAutoencoder(**params)

        # Keep track of losses
        self.loss_tracker = krs.metrics.Mean(name="loss") # total loss
        self.r_loss_tracker = krs.metrics.Mean(name="r_loss") # Reconstruction loss
        self.h_loss_tracker = krs.metrics.Mean(name="h_loss") # Cross-encoding reconstruction loss
        self.kl_loss_tracker = krs.metrics.Mean(name="kl_loss") # Transcoding kullback-leibler loss
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.r_loss_tracker,
            self.h_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def encode(self, I, Xto=True):
        if Xto:
            mu, logsigma2 = self.coderX.encode(I)
        else:
            mu, logsigma2 = self.coderY.encode(I)
        return mu, logsigma2
    
    def reparametrize(self, mu, logsigma2):
        z = tf.random.normal(shape=logsigma2.shape[1:])
        h = mu + tf.exp(tf.clip_by_value(logsigma2, 0., 10.)) * z
        return h
    
    def decode(self, h, toY=True):
        if toY:
            O = self.coderY.decode(h)
        else:
            O = self.coderX.decode(h)
        return O
    
    def transcode(self, I, XtoY=True):
        mu, logsigma2 = self.encode(I, Xto=XtoY)
        h = self.reparametrize(mu, logsigma2)
        O = self.decode(h, toY=XtoY)
        return O
    
    def call(self, data):
        X, Y = data
        Y_fake = self.transcode(X, XtoY=True)
        X_fake = self.transcode(Y, XtoY=False)
        return Y_fake, X_fake

    def compute_r_loss(self, X, X_hat, Y, Y_hat):
        rX_loss = tf.reduce_mean(tf.abs(X - X_hat))
        rY_loss = tf.reduce_mean(tf.abs(Y - Y_hat))
        r_loss = (rX_loss + rY_loss) / 2
        self.r_loss_tracker.update_state(r_loss)
        return r_loss    

    def compute_h_loss(self, muX, logsigma2X, muY, logsigma2Y, muX_hat, logsigma2X_hat, muY_hat, logsigma2Y_hat):
        # hX loss
        muX_loss = tf.reduce_mean(tf.abs(muX - muX_hat))
        #logsigma2X_loss = tf.reduce_mean(tf.abs(logsigma2X - logsigma2X_hat))
        hX_loss = muX_loss #+ logsigma2X_loss
        # hY loss
        muY_loss = tf.reduce_mean(tf.abs(muY - muY_hat))
        #logsigma2Y_loss = tf.reduce_mean(tf.abs(logsigma2Y - logsigma2Y_hat))
        hY_loss = muY_loss #+ logsigma2Y_loss
        # h loss
        h_loss = (hX_loss + hY_loss) / 2
        self.h_loss_tracker.update_state(h_loss)
        return h_loss

    def compute_kl_loss(self, muX, logsigma2X, muY, logsigma2Y):
        # Kullback leibler divergence for each contribution
        klX_loss = tf.reduce_mean(1/2 * (tf.exp(tf.clip_by_value(logsigma2X, 0., 10.)) + muX**2 - logsigma2X - 1))
        klY_loss = tf.reduce_mean(1/2 * (tf.exp(tf.clip_by_value(logsigma2Y, 0., 10.)) + muY**2 - logsigma2Y - 1))
        # Balance individual divergences with relative divergence
        kl_loss = (klX_loss + klY_loss) / 2 #+ tf.abs(klX_loss - klY_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return kl_loss


    def compute_loss(self, X, Y, muX, logsigma2X, muY, logsigma2Y, X_hat, Y_hat, X_fake, Y_fake, muX_hat, logsigma2X_hat, muY_hat, logsigma2Y_hat):
        # Compute loss contributions
        r_loss = self.compute_r_loss(X, X_hat, Y, Y_hat)
        h_loss = self.compute_h_loss(muX, logsigma2X, muY, logsigma2Y, muX_hat, logsigma2X_hat, muY_hat, logsigma2Y_hat)
        kl_loss = self.compute_kl_loss(muX, logsigma2X, muY, logsigma2Y)

        # Compute total loss
        loss = r_loss + self.h_reg * h_loss + self.kl_reg * kl_loss
        self.loss_tracker.update_state(loss)
        return loss
    
    def train_step(self, data):
        # Unpack data
        X, Y = data

        # Forward propagation
        with tf.GradientTape() as tape:
            # Input encoding
            muX, logsigma2X = self.encode(X, Xto=True)
            muY, logsigma2Y = self.encode(Y, Xto=False)

            # Reparametrization
            hX = self.reparametrize(muX, logsigma2X)
            hY = self.reparametrize(muY, logsigma2Y)

            # Self Decoding
            X_hat = self.decode(hX, toY=False)
            Y_hat = self.decode(hY, toY=True)

            # Cross decoding
            Y_fake = self.decode(hX, toY=True)
            X_fake = self.decode(hY, toY=False)

            # Cross Encoding
            muX_hat, logsigma2X_hat = self.encode(Y_fake, Xto=False)
            muY_hat, logsigma2Y_hat = self.encode(X_fake, Xto=True)

            # Compute loss
            loss = self.compute_loss(X, Y, muX, logsigma2X, muY, logsigma2Y, X_hat, Y_hat, X_fake, Y_fake, muX_hat, logsigma2X_hat, muY_hat, logsigma2Y_hat)

        # Backpropagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, self.trainable_weights)

        # Optimization step
        self.optimizer.apply_gradients(zip(gradients,  trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack data
        X, Y = data

        # Input encoding
        muX, logsigma2X = self.encode(X, Xto=True)
        muY, logsigma2Y = self.encode(Y, Xto=False)

        # Reparametrization
        hX = self.reparametrize(muX, logsigma2X)
        hY = self.reparametrize(muY, logsigma2Y)

        # Self Decoding
        X_hat = self.decode(hX, toY=False)
        Y_hat = self.decode(hY, toY=True)

        # Cross decoding
        Y_fake = self.decode(hX, toY=True)
        X_fake = self.decode(hY, toY=False)

        # Cross Encoding
        muX_hat, logsigma2X_hat = self.encode(Y_fake, Xto=False)
        muY_hat, logsigma2Y_hat = self.encode(X_fake, Xto=True)

        # Compute loss
        loss = self.compute_loss(X, Y, muX, logsigma2X, muY, logsigma2Y, X_hat, Y_hat, X_fake, Y_fake, muX_hat, logsigma2X_hat, muY_hat, logsigma2Y_hat)

        return {m.name: m.result() for m in self.metrics}

@krs.saving.register_keras_serializable()
class GANDiscriminators(krs.models.Model):
    def __init__(self, **params):
        super().__init__()

        # Initialize two identical discriminators
        self.discriminatorX = GANDiscriminator(**params)
        self.discriminatorY = GANDiscriminator(**params)

        self.d_loss_tracker = krs.metrics.Mean(name="d_loss")
    
    @property
    def metrics(self):
        return [self.d_loss_tracker]

    def compute_loss(self, pX_real, pX_fake, pY_real, pY_fake):
        # X loss
        realX_loss = tf.reduce_mean(((pX_real - 1) - tf.reduce_mean(pX_fake))**2)
        fakeX_loss = tf.reduce_mean((pX_fake - (tf.reduce_mean(pX_real) - 1))**2)
        dX_loss = (realX_loss + fakeX_loss) / 2
        # Y loss
        realY_loss = tf.reduce_mean(((pY_real - 1) - tf.reduce_mean(pY_fake))**2)
        fakeY_loss = tf.reduce_mean((pY_fake - (tf.reduce_mean(pY_real) - 1))**2)
        dY_loss = (realY_loss + fakeY_loss) / 2
        # Combined
        d_loss = (dX_loss + dY_loss) / 2
        self.d_loss_tracker.update_state(d_loss)
        return d_loss
    
    def call(self, data):
        # Unpack data
        X, Y = data
        pX = self.discriminatorX(X)
        pY = self.discriminatorY(Y)
        return pX, pY
    
    def train_step(self, data, generator):
        # Unpack data
        X, Y = data

        # Encode data
        contentX, styleX = generator.encode(X, Xto=True)
        contentY, styleY = generator.encode(Y, Xto=False)

        # Generate fake styles
        styleX_fake = generator.sample_style(styleX)
        styleY_fake = generator.sample_style(styleY)

        # Cross decode
        Y_fake = generator.decode(contentX, styleY_fake, toY=True)
        X_fake = generator.decode(contentY, styleX_fake, toY=False)

        # Forward propagation
        with tf.GradientTape() as tape:
            # Discriminate
            pX_real = self.discriminatorX(X)
            pX_fake = self.discriminatorX(X_fake)
            pY_real = self.discriminatorY(Y)
            pY_fake = self.discriminatorY(Y_fake)

            # Compute loss
            loss = self.compute_loss(pX_real, pX_fake, pY_real, pY_fake)
        
        # Backpropagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, self.trainable_weights)

        # Optimization step
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data, generator):
        # Unpack data
        X, Y = data

        # Encode data
        contentX, styleX = generator.encode(X, Xto=True)
        contentY, styleY = generator.encode(Y, Xto=False)

        # Generate fake styles
        styleX_fake = generator.sample_style(styleX)
        styleY_fake = generator.sample_style(styleY)

        # Cross decode
        Y_fake = generator.decode(contentX, styleY_fake, toY=True)
        X_fake = generator.decode(contentY, styleX_fake, toY=False)

        # Discriminate
        pX_real = self.discriminatorX(X)
        pX_fake = self.discriminatorX(X_fake)
        pY_real = self.discriminatorY(Y)
        pY_fake = self.discriminatorY(Y_fake)

        # Compute loss
        loss = self.compute_loss(pX_real, pX_fake, pY_real, pY_fake)

        # Return loss
        return {m.name: m.result() for m in self.metrics}

@krs.saving.register_keras_serializable()
class GANTranscoder(krs.models.Model):
    def __init__(self, **params):
        super().__init__()

        # Pop transcoder params
        self.gan_reg = params.get("gan_reg")
        self.c_reg = params.get("c_reg")
        self.s_reg = params.get("s_reg")
        self.use_fake_style = params.pop("use_fake_style")
        self.is_munit = params.pop("is_munit", False)
        
        # Intialize two identical autoencoders
        if self.is_munit:
            self.coderX = MUNITGenerator(**params)
            self.coderY = MUNITGenerator(**params)
        else:
            self.coderX = GANGenerator(**params)
            self.coderY = GANGenerator(**params)

        # Keep track of losses
        self.loss_tracker = krs.metrics.Mean(name="loss") # Total loss
        self.r_loss_tracker = krs.metrics.Mean(name="r_loss") # Reconstruction loss
        self.gan_loss_tracker = krs.metrics.Mean(name="gan_loss") # Adverserial loss
        self.c_loss_tracker = krs.metrics.Mean(name="c_loss") # Content loss
        self.s_loss_tracker = krs.metrics.Mean(name="s_loss") # Style loss
    
    @property
    def metrics(self):
        return [
            self.loss_tracker, 
            self.r_loss_tracker,
            self.gan_loss_tracker, 
            self.c_loss_tracker, 
            self.s_loss_tracker
        ]

    def encode(self, I, Xto=True):
        if Xto:
            content, style = self.coderX.encode(I)
        else:
            content, style = self.coderY.encode(I)
        return content, style

    def sample_style(self, style):
        style_fake = self.coderX.sample_style(style)
        return style_fake

    def decode(self, content, style, toY=True):
        if toY:
            O = self.coderY.decode(content, style)
        else:
            O = self.coderX.decode(content, style)
        return O

    def transcode(self, I, XtoY=True):
        content, style = self.encode(I, Xto=XtoY)
        style_fake = self.sample_style(style)
        O = self.decode(content, style_fake, toY=XtoY)
        return O
    
    def call(self, data):
        X, Y = data
        # Transcode in both directions
        Y_fake = self.transcode(X, XtoY=True)
        X_fake = self.transcode(Y, XtoY=False)
        return Y_fake, X_fake

    def compute_r_loss(self, X, X_hat, Y, Y_hat):
        rX_loss = tf.reduce_mean(tf.abs(X - X_hat))
        rY_loss = tf.reduce_mean(tf.abs(Y - Y_hat))
        r_loss = (rX_loss + rY_loss) / 2
        self.r_loss_tracker.update_state(r_loss)
        return r_loss
    
    def compute_gan_loss(self, pX_real, pX_fake, pY_real, pY_fake):
        # Gan loss for X
        realX_loss = tf.reduce_mean((pX_real - (tf.reduce_mean(pX_fake) - 1))**2)
        fakeX_loss = tf.reduce_mean(((pX_fake - 1) - tf.reduce_mean(pX_real))**2)
        ganX_loss = (realX_loss + fakeX_loss) / 2
        # Gan loss for Y
        realY_loss = tf.reduce_mean((pY_real - (tf.reduce_mean(pY_fake) - 1))**2)
        fakeY_loss = tf.reduce_mean(((pY_fake - 1) - tf.reduce_mean(pY_real))**2)
        ganY_loss = (realY_loss + fakeY_loss) / 2
        # Total Gan loss
        gan_loss = (ganX_loss + ganY_loss) / 2
        self.gan_loss_tracker.update_state(gan_loss)
        return gan_loss
    
    def compute_c_loss(self, contentX, contentX_hat, contentY, contentY_hat):
        cX_loss = tf.reduce_mean(tf.abs(contentX - contentX_hat))
        cY_loss = tf.reduce_mean(tf.abs(contentY - contentY_hat))
        c_loss = (cX_loss + cY_loss) / 2
        self.c_loss_tracker.update_state(c_loss)
        return c_loss

    def compute_s_loss(self, styleX_fake, styleX_fake_hat, styleY_fake, styleY_fake_hat):
        sX_loss = tf.reduce_mean(tf.abs(styleX_fake - styleX_fake_hat))
        sY_loss = tf.reduce_mean(tf.abs(styleY_fake - styleY_fake_hat))
        s_loss = (sY_loss + sY_loss) / 2
        self.s_loss_tracker.update_state(s_loss)
        return s_loss

    def compute_loss(
        self,
        X, Y,
        contentX, styleX, contentY, styleY,
        X_hat, Y_hat,
        styleX_fake, styleY_fake,
        X_fake, Y_fake,
        pX_real, pX_fake, pY_real, pY_fake,
        contentX_hat, contentY_hat,
        styleX_fake_hat, styleY_fake_hat,
    ):
        # Compute loss contributions
        r_loss = self.compute_r_loss(X, X_hat, Y, Y_hat)
        gan_loss = self.compute_gan_loss(pX_real, pX_fake, pY_real, pY_fake)
        c_loss = self.compute_c_loss(contentX, contentX_hat, contentY, contentY_hat)
        s_loss = self.compute_s_loss(styleX_fake, styleX_fake_hat, styleY_fake, styleY_fake_hat)

        # Copute total loss
        loss = r_loss + self.gan_reg * gan_loss + self.c_reg * c_loss + self.s_reg * s_loss
        self.loss_tracker.update_state(loss)
        return loss

    def train_step(self, data, discriminator):
        # Unpack data
        X, Y = data

        # Forward propagation
        with tf.GradientTape() as tape:
            # Encode
            contentX, styleX = self.encode(X, Xto=True)
            contentY, styleY = self.encode(Y, Xto=False)

            # Self Decode
            X_hat = self.decode(contentX, styleX, toY=False)
            Y_hat = self.decode(contentY, styleY, toY=True)

            # Generate fake styles
            styleX_fake = self.sample_style(styleX) if self.use_fake_style else styleX
            styleY_fake = self.sample_style(styleY) if self.use_fake_style else styleY

            # Cross Decode
            Y_fake = self.decode(contentX, styleY_fake, toY=True)
            X_fake = self.decode(contentY, styleX_fake, toY=False)

            # Discriminate
            pX_real = discriminator.discriminatorX(X)
            pX_fake = discriminator.discriminatorX(X_fake)
            pY_real = discriminator.discriminatorY(Y)
            pY_fake = discriminator.discriminatorY(Y_fake)

            # Cross Encode
            contentX_hat, styleY_fake_hat = self.encode(Y_fake, Xto=False)
            contentY_hat, styleX_fake_hat = self.encode(X_fake, Xto=True)

            # Compute Loss
            loss = self.compute_loss(
                X, Y,
                contentX, styleX, contentY, styleY,
                X_hat, Y_hat,
                styleX_fake, styleY_fake,
                X_fake, Y_fake,
                pX_real, pX_fake, pY_real, pY_fake,
                contentX_hat, contentY_hat,
                styleX_fake_hat, styleY_fake_hat,
            )
        
        # Backpropagation
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, self.trainable_weights)

        # Optimization step
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Return loss
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data, discriminator):
        # Unpack data
        X, Y = data

        # Encode
        contentX, styleX = self.encode(X, Xto=True)
        contentY, styleY = self.encode(Y, Xto=False)

        # Self Decode
        X_hat = self.decode(contentX, styleX, toY=False)
        Y_hat = self.decode(contentY, styleY, toY=True)

        # Generate fake styles
        styleX_fake = self.sample_style(styleX) if self.use_fake_style else styleX
        styleY_fake = self.sample_style(styleY) if self.use_fake_style else styleY

        # Cross Decode
        Y_fake = self.decode(contentX, styleY_fake, toY=True)
        X_fake = self.decode(contentY, styleX_fake, toY=False)

        # Discriminate
        pX_real = discriminator.discriminatorX(X)
        pX_fake = discriminator.discriminatorX(X_fake)
        pY_real = discriminator.discriminatorY(Y)
        pY_fake = discriminator.discriminatorY(Y_fake)

        # Cross Encode
        contentX_hat, styleY_fake_hat = self.encode(Y_fake, Xto=False)
        contentY_hat, styleX_fake_hat = self.encode(X_fake, Xto=True)

        # Compute Loss
        loss = self.compute_loss(
            X, Y,
            contentX, styleX, contentY, styleY,
            X_hat, Y_hat,
            styleX_fake, styleY_fake,
            X_fake, Y_fake,
            pX_real, pX_fake, pY_real, pY_fake,
            contentX_hat, contentY_hat,
            styleX_fake_hat, styleY_fake_hat,
        )

        # Return loss
        return {m.name: m.result() for m in self.metrics}

