# %% Import libraries
import os
import shutil
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

# %% Import tools
from tools.audio_tools import read_audio, play_audio, sample_rate
from tools.feature_tools import compute_stft, compute_istft, compute_mels, compute_imels, compute_mfcc, compute_imfcc
from tools.plot_tools import make_figax

# %% Define data directories
classical_dir = "classical"
jazz_dir = "jazz"
cut_data_path = os.path.join(".", "raw_data_cuts")
cut_classical_path = os.path.join(cut_data_path, classical_dir)
cut_jazz_path = os.path.join(cut_data_path, jazz_dir)


# %% Define genres and feature spaces
space_labels = ["STFT", "MELS", "MFCC"]
space_transforms = [compute_stft, compute_mels, compute_mfcc]
space_inverse_transforms = [compute_istft, compute_imels, compute_imfcc]

genre_labels = ["CLASSICAL", "JAZZ"]
genre_paths = [cut_classical_path, cut_jazz_path]


# %% Explore feature spaces for all genres
for genre_label, genre_path in zip(genre_labels, genre_paths):
    print(genre_label)

    # Pick a random audio track
    audio_file = np.random.choice(os.listdir(genre_path))
    audio_path = os.path.join(genre_path, audio_file)
    audio = read_audio(audio_path)
    time = np.linspace(0, len(audio)/sample_rate, len(audio))
    print(f"{audio_file = }")
    print(f"{len(audio) = }")

    # Time domain plot
    fig, ax = make_figax()
    ax.plot(time, audio)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Audio Track")
    ax.set_title(f"{genre_label} AUDIO TRACK")
    plt.show()

    # Play audio
    print(f"{genre_label} PLAYER")
    player = play_audio(audio)

    # Transform to feature spaces and make plots
    for space_label, space_transform, space_inverse_transform in zip(space_labels, space_transforms, space_inverse_transforms):
        print(space_label)

        # Compute transform
        transform = space_transform(audio)
        print(f"{transform.shape = }")
        print(f"{transform.size = }")
        print(f"transform element = {np.random.choice(transform.reshape(-1))}")

        # Plot Transform in 2D
        fig, ax = make_figax()
        if space_label == "STFT":
            audio_plot = lb.amplitude_to_db(np.abs(transform))
        elif space_label == "MELS":
            audio_plot = lb.power_to_db(transform)
        else:
            audio_plot = 20*np.log10(np.abs(transform))
        img = ax.imshow(audio_plot, origin="lower", aspect="auto")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Transform index")
        ax.set_title(space_label)
        plt.colorbar(img, ax=ax)
        plt.show()

        # Plot transform coefficient roloff
        transform_magnitudes = np.sort(audio_plot.reshape(-1))[::-1]
        fig, ax = make_figax()
        ax.plot(transform_magnitudes)
        ax.set_xlabel("Transform coefficient")
        ax.set_ylabel("Transform magnitude [dB]")
        ax.set_title(space_label)
        plt.show()

        # Transform reconstruct & play
        print(f"INVERSE {space_label} - {genre_label}")
        inverse = space_inverse_transform(transform)
        play_audio(inverse)

        # Transform reconstruct from top k & play
        print(f"INVERSE {space_label} - {genre_label} - TOP K")
        fraction = 1/8
        k = int(transform.size * fraction)
        print(f"Reconstructing from 1/{1/fraction:.2f} of the samples: {k = }")
        transform[audio_plot <= transform_magnitudes[k]] = 0.
        inverse = space_inverse_transform(transform)
        play_audio(inverse)

# %%
