import librosa as lb
import numpy as np
import sys
import os
from .constants import default_n_fft, default_n_mels, default_sample_rate, default_n_mfcc
from .audio_tools import read_audio

# Domain transform helper functions
def pad_signal(x, n_fft = default_n_fft):
    x_padded = lb.util.fix_length(x, size=len(x) + n_fft // 2)
    return x_padded

def compute_stft(x, n_fft = default_n_fft, **kwargs):
    x_padded = pad_signal(x, n_fft=n_fft)
    x_stft = lb.stft(x_padded, n_fft=n_fft, **kwargs)
    return np.transpose(x_stft)

def compute_istft(x_stft, n_fft = default_n_fft, **kwargs):
    x_istft = lb.istft(np.transpose(x_stft), n_fft=n_fft, **kwargs)
    return x_istft

def compute_mels(x, n_fft = default_n_fft, n_mels = default_n_mels, sr = default_sample_rate, **kwargs):
    x_padded = pad_signal(x, n_fft=n_fft)
    x_mels = lb.feature.melspectrogram(y=x_padded, sr=sr, n_mels=n_mels, n_fft=n_fft, **kwargs)
    return np.transpose(x_mels)

def compute_imels(x_mels, n_fft = default_n_fft, sr = default_sample_rate, **kwargs):
    x_imels = lb.feature.inverse.mel_to_audio(np.transpose(x_mels), sr=sr, n_fft=n_fft, **kwargs)
    return x_imels

def compute_mfcc(x, sr=default_sample_rate, n_fft = default_n_fft, n_mels = default_n_mels, n_mfcc = default_n_mfcc, **kwargs):
    x_padded = pad_signal(x, n_fft=n_fft)
    x_mfcc = lb.feature.mfcc(y=x_padded, sr=sr, n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc, **kwargs)
    return np.transpose(x_mfcc)

def compute_imfcc(x_mfcc, sr=default_sample_rate, n_fft = default_n_fft, n_mels = default_n_mels, **kwargs):
    x_imfcc = lb.feature.inverse.mfcc_to_audio(np.transpose(x_mfcc), sr=sr, n_mels=n_mels, n_fft=n_fft, **kwargs)
    return x_imfcc

def load_data(data_path, feature_extractors = [compute_mels], n_samples="all"):
    # Read directory
    samples = os.listdir(data_path)
    # Sample files
    if n_samples != "all":
        n_samples = min(len(samples), n_samples)
        samples = np.random.choice(samples, size=n_samples, replace=False)
    else:
        n_samples = len(samples)
    # Load files
    data = [None] * n_samples
    for i, sample_file in enumerate(samples):
        # Read audio
        audio = read_audio(os.path.join(data_path, sample_file))
        # Extract features
        features = np.array([extractor(audio) for extractor in feature_extractors])
        # Add features to data
        data[i] = features
        # Print progress
        if i % (n_samples // 100) == 0:
            sys.stdout.write('\r')
            sys.stdout.write(f"{(i+1) / n_samples * 100:.0f} %")
            sys.stdout.flush()
    print()
    data = np.transpose(np.array(data), axes=[0, 2, 3, 1])
    # Return extracted features
    return data

# Save normalization factors globally
X_dev = np.NaN
X_weights = np.NaN

def normalize_features(X_raw_train, X_raw_val = None, X_raw_test = None, epsilon=1e-3):
    global X_dev, X_weights
    # Rescale by the standard deviation
    X_dev = np.std(X_raw_train)
    X_train = X_raw_train / X_dev
    
    # Rescale by Logarithm
    X_train = np.log10(1 + X_train)

    # Frequency weight scaling
    X_weights = np.std(X_train, axis=(0,1,3)).reshape(1,1,-1,1) + epsilon
    X_train /= X_weights

    # Apply to test set
    if X_raw_val is not None:
        X_val = X_raw_val / X_dev
        X_val = np.log10(X_val + 1)
        X_val /= X_weights

    if X_raw_test is not None:
        X_test = X_raw_test / X_dev
        X_test = np.log10(X_test + 1)
        X_test /= X_weights
    
    if X_raw_test is None and X_raw_val is None:
        return X_train
    elif X_raw_test is None and X_raw_val is not None:
        return X_train, X_val
    elif X_raw_test is not None and X_raw_val is None:
        return X_train, X_test
    else:
        return X_train, X_val, X_test
   

def denormalize_features(X):
    global X_dev, X_weights
    X_raw = (10**(X * X_weights) - 1) * X_dev
    return X_raw