# Constants
import os

## Data dirs
data_path = "./"
raw_data_path = os.path.join(data_path, "raw_data")
raw_classical_path = os.path.join(raw_data_path, "classical")
raw_jazz_path = os.path.join(raw_data_path, "jazz")
cut_data_path = os.path.join(data_path, "cut")
cut_classical_path = os.path.join(cut_data_path, "classical")
cut_jazz_path = os.path.join(cut_data_path, "jazz")
models_dir = "models"
models_path = os.path.join(data_path, models_dir)

# Preprocessing settings
n_secs = 1
cut_classical_path = os.path.join(cut_classical_path, f"{n_secs}s")
cut_jazz_path = os.path.join(cut_jazz_path, f"{n_secs}s")

## Domain sizes
default_sample_rate = 44100
default_n_fft = 2048
default_n_mels = 128
default_n_mfcc = 128