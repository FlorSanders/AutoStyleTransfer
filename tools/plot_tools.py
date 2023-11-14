import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("tableau-colorblind10")
from .constants import default_sample_rate

# Ploting helper functions
def make_figax(n_h=1, n_v=1, w_fig=8, h_fig=4.5):
    fig, ax = plt.subplots(n_v, n_h, figsize=(w_fig * n_h, h_fig * n_v))
    return fig, ax

def style_figax(fig, ax, grid = True):
    ax.grid(grid)
    fig.tight_layout()
    return fig, ax

def plot_audio(x, fn=None, sr=default_sample_rate, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = make_figax()

    if fn is None:
        fn = lambda x: x

    time = np.linspace(0, len(x) / sr, len(x))
    ax.plot(time, fn(x))
    ax.set_xlabel("Time")
    ax.set_ylabel("Audio")

    return fig, ax

def plot_spectral_feature(feature, fn=lambda x: 20 * np.log10(x + 1e-3), fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = make_figax()

    if fn is None:
        fn = lambda x: x

    img = ax.imshow(np.transpose(fn(feature)), origin="lower", aspect="auto", interpolation="none")
    plt.colorbar(img, ax=ax)

    return fig, ax

def plot_history(history, fig = None, ax = None):
    if not isinstance(history, dict):
        history = history.history
    loss = history["loss"]
    val_loss = history["val_loss"]
    
    if fig is None or ax is None:
        fig, ax = make_figax()
    
    ax.plot(loss, label="Training Loss")
    ax.plot(val_loss, label="Validation Loss")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    
    return fig, ax