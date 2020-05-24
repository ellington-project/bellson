# various global configuration options that may be overridden by main.
# Where we should write cached histograms
cache_directory = "/tmp"

# Speedups/slowdown variants for tracks to augment out training set.
augmentation_variants = [0.75, 0.8, 0.85,
                         0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25]

# This controls how "big" our frequency range is for training.
SPECTROGRAM_LOW_FREQUENCY_CUTOFF = 64
SPECTROGRAM_HIGH_FREQUENCY_CUTOFF = 320

# The shapes of the resulting input tensors
# Note, input_time_dim is user adjustable, but input_freq_dim depends on the cutoff frequencies that we specify.
input_time_dim = 1720
input_freq_dim = SPECTROGRAM_HIGH_FREQUENCY_CUTOFF - \
    SPECTROGRAM_LOW_FREQUENCY_CUTOFF

# Parameters for librosa.core.stft. Manually specified so that we can keep track of them.
SAMPLE_RATE = 44100
N_FFT = 1024
WIN_LENGTH = N_FFT
HOP_LENGTH = int(WIN_LENGTH/4)
