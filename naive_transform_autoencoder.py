# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as krs
import os
from sklearn.model_selection import train_test_split

# %% Import helpers
from tools.audio_tools import read_audio, play_audio
from tools.feature_tools import compute_mels, compute_imels, sample_rate_default, n_fft_default, n_mels_default
from tools.plot_tools import make_figax, plot_audio, plot_spectral_feature
from tools.tensorflow_tools import TransformLayer

# %% Load dataset
n_samples = 1000

# Read files
print("Reading Data Directory")
data_dir = os.path.join(".", "raw_data_cuts", "classical")
data_files = os.listdir(data_dir)
print(f"{len(data_files) = }")

# Make a random selection
print("Making Sample Selection")
sample_files = np.random.choice(data_files, size=n_samples, replace=False) if len(data_files) > n_samples else data_file

# Read audio tracks
print("Reading Audio Tracks")
sample_audio = np.array([read_audio(os.path.join(data_dir, audio_file)) for audio_file in sample_files])

# %% Make the test/train split
train_features, test_features = train_test_split(sample_audio, test_size=0.2)
print(train_features.shape)
print(f"{len(train_features) = }")
print(f"{len(test_features) = }")

# %% Define Transform -> Inverse Model 
class NaiveTransformAutoencoder(krs.models.Model):
    def __init__(self, latent_dim, transform_shape, **kwargs):
        super().__init__()
        self.transform_shape = transform_shape
        self.transform_size = np.multiply.reduce(transform_shape)
        self.latent_dim = latent_dim

        self.encoder = krs.Sequential([
            TransformLayer(compute_mels, **kwargs),
            krs.layers.Flatten(),
            krs.layers.Dense(self.latent_dim, activation="relu")
        ])

        self.decoder = krs.Sequential([
            krs.layers.Dense(self.transform_size, activation="relu"),
            krs.layers.Reshape(self.transform_shape),
            TransformLayer(compute_imels, **kwargs)
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# %% Compute transform shape
audio_length_seconds = 1
sample_length = audio_length_seconds * sample_rate_default
padded_length = sample_length + n_fft_default // 2
transform_frames = int(padded_length / n_fft_default * 4) + 1
transform_shape = (n_mels_default, transform_frames)
transform_size = np.multiply.reduce(transform_shape)

# %% Construct the autoencoder
latent_dim = transform_size // 8
autoencoder = NaiveTransformAutoencoder(latent_dim, transform_shape)
autoencoder.compile(optimizer="adam", loss=krs.losses.MeanSquaredLogarithmicError())
# autoencoder.build(sample_length)
# autoencoder.summary()

# %% Train the autoencoder
autoencoder.fit(train_features, train_features, epochs=len(train_features) // 10, shuffle=True, validation_data=(test_features, test_features))

# %% Test the autoencoder
n_tests = 1
test_files = np.random.choice(data_files, size=n_tests)

for test_file in test_files:
    print(f"File = {test_file}\n---")
    print("Original audio")
    audio = read_audio(os.path.join(data_dir, test_file))
    fig, ax = plot_audio(audio)
    ax.set_title("Original Audio")
    plt.show()
    player = play_audio(audio)
    
    print("Extracting feature")
    audio_feature = compute_mels(audio).astype("float32")
    fig, ax = plot_spectral_feature(audio_feature)
    ax.set_title("Feature space")
    plt.show()

    print("Reconstructing audio")
    inverse_audio = compute_imels(audio_feature)
    fig, ax = plot_audio(inverse_audio)
    ax.set_title("Inverted Audio")
    plt.show()
    player = play_audio(inverse_audio)

    print("Reconstructing audio")
    reconstructed_audio = np.array(autoencoder.call(np.array([audio])))[0].astype(float)
    fig, ax = plot_audio(reconstructed_audio)
    ax.set_title("Reconstructed Audio")
    plt.show()
    player = play_audio(reconstructed_audio)
    
    print()


# %%
