"""
Plotting utilities for models and data.
"""
import matplotlib.pyplot as plt

def plot_model(model, ax=None, cmap='viridis'):
    """
    Show a 2D model (e.g., velocity or resistivity) on a given axis.
    """
    if ax is None:
        fig, ax = plt.subplots()
    cax = ax.imshow(model, origin='lower', cmap=cmap, aspect='auto')
    plt.colorbar(cax, ax=ax, label='Value')
    return ax

def plot_data_section(data, ax=None):
    """
    Plot synthetic data (e.g. shot gather or section).
    """
    if ax is None:
        fig, ax = plt.subplots()
    cax = ax.imshow(data, aspect='auto', cmap='seismic', origin='lower')
    plt.colorbar(cax, ax=ax, label='Amplitude')
    return ax
