# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split

# %% Import helpers
from tools.audio_tools import read_audio, play_audio
from tools.feature_tools import compute_mels, compute_imels
from tools.plot_tools import make_figax, plot_audio, plot_spectral_feature

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

# Extract the features
print("Extracting Features")
sample_features = np.array([compute_mels(audio) for audio in sample_audio]).astype("float32")
feature_shape = sample_features[0].shape
feature_size = sample_features[0].size
print(f"{feature_shape = }")
print(f"{feature_size = }")
sample_features.reshape((len(sample_features), -1))

# %% Make the test/train split
train_features, test_features = train_test_split(sample_features, test_size=0.2)
print(train_features.shape)
print(f"{len(train_features) = }")
print(f"{len(test_features) = }")

# %% Define the autoencoder model
class Autoencoder1(Model):
    def __init__(self, latent_dim, feature_shape):
        super().__init__()

        # Saving the feature & latent space dimensions
        self.latent_dim = latent_dim
        self.feature_shape = feature_shape
        self.feature_size = np.multiply.accumulate(feature_shape)[-1]

        # Building encoder & decoder models (single layer)
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(self.latent_dim, activation="linear"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(self.feature_size, activation="sigmoid"),
            layers.Reshape(feature_shape)
        ])

    def call(self, x):
        scale = tf.math.reduce_max(x)
        encoded = self.encoder(tf.math.divide(x, scale))
        decoded = self.decoder(encoded)
        return tf.math.multiply(decoded, scale)

# %% Create an autoencoder
latent_dim = feature_size // 4
autoencoder = Autoencoder1(latent_dim, feature_shape)
autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredLogarithmicError())

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


    print("Autoencoding")
    reconstructed_feature = np.array(autoencoder.call(np.array([audio_feature])))[0].astype(float)
    fig, ax = plot_spectral_feature(reconstructed_feature)
    ax.set_title("Autoencoded feature space")
    plt.show()

    print("Reconstructing autencoded audio")
    reconstructed_audio = compute_imels(reconstructed_feature)
    fig, ax = plot_audio(reconstructed_audio)
    ax.set_title("Reconstructed Audio")
    plt.show()
    player = play_audio(reconstructed_audio)
    
    print()


# %%
