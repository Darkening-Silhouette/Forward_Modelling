"""
Compute detectability metrics for synthetic data.
"""
import numpy as np

def signal_to_noise(signal, noise):
    """
    Compute signal-to-noise ratio (peak signal / std of noise).
    """
    if noise is None or np.std(noise) == 0:
        return np.inf
    return np.max(np.abs(signal)) / np.std(noise)

def depth_of_investigation(data, threshold=0.5):
    """
    Rough estimation: depth index at which amplitude falls below threshold fraction of max.
    (Placeholder implementation.)
    """
    amp = np.max(np.abs(data), axis=1)
    max_amp = np.max(amp)
    idx = np.where(amp < threshold * max_amp)[0]
    return idx[0] if len(idx) > 0 else len(amp)
