import matplotlib.pyplot as plt
import numpy as np
from .audio_tools import sample_rate

def make_figax(n_h=1, n_v=1, w_fig=8, h_fig=4.5):
    fig, ax = plt.subplots(n_v, n_h, figsize=(w_fig * n_h, h_fig * n_v))
    return fig, ax

def style_figax(fig, ax, grid = True):
    ax.grid(grid)    
    plt.tight_layout()

def plot_audio(x, fn=None, sr=sample_rate, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = make_figax()
    
    if fn is None:
        fn = lambda x: x

    time = np.linspace(0, len(x) / sample_rate, len(x))
    ax.plot(time, fn(x))
    ax.set_xlabel("Time")
    ax.set_ylabel("Audio")

    return fig, ax

def plot_spectral_feature(feature, fn=lambda x: 20 * np.log10(x), fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = make_figax()
    
    if fn is None:
        fn = lambda x: x
    
    img = ax.imshow(fn(feature), origin="lower", aspect="auto")
    plt.colorbar(img, ax=ax)

    return fig, ax