# AutoStyleTransfer

An investigation into music style transfer with twin autoencoders.

## Instructions to run the code

In order to run this repository, the following prerequisites are to be installed:

- `python==3.10`
- `numpy==1.25.2`
- `tensorflow==2.14.0`
- `tensorflow_probability==0.22.1`
- `scikit-learn==1.3.1`
- `librosa==0.10.1`

The repository consists of five jupyter notebooks containing the implementation.

- `data_preprocessing.ipynb`: Reads the raw audio files, normalizes them, cuts into segments, extracts features (optional) and saves to disk.
- `data_exploration.ipynb`: Reads the cut audio files and explores different feature transforms and their properties.
- `autoencoders.ipynb`: Trains the different autoencoder models in order to identify the optimal hyperparameters for the transcoder subnetworks.
- `transcoders.ipynb`: Trains and evaluates the style transfer transcoders.
- `demo.ipynb`: Loads the weights of the pretrained style transfer transcoders and performs style transfer for the demo audio samples.

The project slides and report are included in the repository.
