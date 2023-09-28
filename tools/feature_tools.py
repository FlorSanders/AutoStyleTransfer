import librosa as lb
from .audio_tools import sample_rate

# Default values
n_fft_default = 2048
n_mels_default = 256
n_mfcc_default = 256
sample_rate_default = sample_rate

# Feature transform functions
def pad_signal(x, n_fft = n_fft_default):
    x_padded = lb.util.fix_length(x, size=len(x) + n_fft // 2)
    return x_padded

def compute_stft(x, n_fft = n_fft_default, **kwargs):
    x_padded = pad_signal(x, n_fft=n_fft)
    x_stft = lb.stft(x_padded, n_fft=n_fft, **kwargs)
    return x_stft

def compute_istft(x_stft, n_fft = n_fft_default, **kwargs):
    x_istft = lb.istft(x_stft, n_fft=n_fft, **kwargs)
    return x_istft

def compute_mels(x, n_fft = n_fft_default, n_mels = n_mels_default, sr = sample_rate_default, **kwargs):
    x_padded = pad_signal(x, n_fft=n_fft)
    x_mels = lb.feature.melspectrogram(y=x_padded, sr=sr, n_mels=n_mels, n_fft=n_fft, **kwargs)
    return x_mels

def compute_imels(x_mels, n_fft = n_fft_default, sr = sample_rate_default, **kwargs):
    x_imels = lb.feature.inverse.mel_to_audio(x_mels, sr=sr, n_fft=n_fft, **kwargs)
    return x_imels

def compute_mfcc(x, sr=sample_rate_default, n_fft = n_fft_default, n_mels = n_mels_default, n_mfcc = n_mfcc_default, **kwargs):
    x_padded = pad_signal(x, n_fft=n_fft)
    x_mfcc = lb.feature.mfcc(y=x_padded, sr=sr, n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc, **kwargs)
    return x_mfcc

def compute_imfcc(x_mfcc, sr=sample_rate_default, n_fft = n_fft_default, n_mels = n_mels_default, **kwargs):
    x_imfcc = lb.feature.inverse.mfcc_to_audio(x_mfcc, sr=sr, n_mels=n_mels, n_fft=n_fft, **kwargs)
    return x_imfcc
