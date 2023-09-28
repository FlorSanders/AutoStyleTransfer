# %% Import libraries
import os
import shutil
import librosa as lb
import soundfile as sf
import numpy as np

# %% Define path names
classical_dir = "classical"
jazz_dir = "jazz"
# Input dirs
raw_data_path = os.path.join(".", "raw_data")
raw_classical_path = os.path.join(raw_data_path, classical_dir)
raw_jazz_path = os.path.join(raw_data_path, jazz_dir)

# Output dirs
cut_data_path = os.path.join(".", "raw_data_cuts")
cut_classical_path = os.path.join(cut_data_path, classical_dir)
cut_jazz_path = os.path.join(cut_data_path, jazz_dir)

# %% OPTIONALLY - DELETE ALL PREPROCESSED DATA
if os.path.exists(cut_data_path):
    shutil.rmtree(cut_data_path)

# %% Create output dirs
os.makedirs(cut_classical_path, exist_ok=True)
os.makedirs(cut_jazz_path, exist_ok=True)

# %% Define cutting function
def cut_audio_file(source_file: str, source_dir: str, target_dir: str, segment_duration: int = 1, target_loudness: float = -20, target_sample_rate = 44100):
    # Debug message
    print(f"Processing file: {source_file}")

    # File name & path
    file_name, file_ext = os.path.splitext(source_file)
    source_path = os.path.join(source_dir, source_file)

    # Read input file (mono @ target_sample_rate)
    audio, sample_rate = lb.load(source_path, sr=target_sample_rate, mono=True)

    # Equalize loudness
    current_rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_loudness/20)
    audio_normalized = audio * target_rms / current_rms

    # Compute the number of samples per segment
    samples_per_segment = int(sample_rate * segment_duration)
    n_segments = int(len(audio_normalized) / samples_per_segment)

    # Split into segments & save to file
    for i in range(n_segments):
        # split segment
        segment = audio_normalized[i*samples_per_segment:(i+1)*samples_per_segment]

        # construct output path
        target_file = f"{file_name}_{i}.wav"
        target_path = os.path.join(target_dir, target_file)

        # Write output
        sf.write(target_path, segment, sample_rate)
    
    return n_segments
        
# %% Example cutting (1 sample for each genre)
for raw_path, cut_path in zip([raw_classical_path, raw_jazz_path], [cut_classical_path, cut_jazz_path]):
    raw_files = os.listdir(raw_path)
    source_file = raw_files[0]
    n_segments = cut_audio_file(source_file, raw_path, cut_path)
    print(f"Cut into {n_segments} segments.")

# %% Bulk cutting (N samples for each genre)
n_samples = 5
for raw_path, cut_path in zip([raw_classical_path, raw_jazz_path], [cut_classical_path, cut_jazz_path]):
    raw_files = os.listdir(raw_path)

    for source_file in raw_files[:n_samples]:
        file_name, file_ext = os.path.splitext(source_file)
        
        if os.path.exists(os.path.join(cut_path, f"{file_name}_0.wav")):
            print(f"File {file_name} was already processed")
            continue
        
        n_segments = cut_audio_file(source_file, raw_path, cut_path)
        print(f"Cut into {n_segments} segments.")


# %%
