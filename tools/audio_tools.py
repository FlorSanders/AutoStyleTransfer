import os
import librosa as lb
import soundfile as sf
from IPython.display import Audio, display

sample_rate = 44100

def read_audio(audio_path: os.PathLike, sr = sample_rate, mono=True, **kwargs):
    audio, _ = lb.load(audio_path, sr=sr, mono=mono, **kwargs)
    return audio

def write_audio(audio, audio_path: os.PathLike, sr = sample_rate, **kwargs):
    sf.write(audio_path, audio, sr, **kwargs)

def play_audio(audio, sr = sample_rate):
    player = Audio(data=audio, rate=sr)
    return display(player)
