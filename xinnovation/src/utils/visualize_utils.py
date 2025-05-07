#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Visualization Module

This module provides functions for visualizing matrices as heatmaps, 
supporting both interactive visualization and saving to PNG files.
It handles both NumPy arrays and PyTorch tensors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

__all__ = [
    'convert_to_numpy',
    'visualize_matrix_interactive',
    'save_matrix_heatmap'
]


def convert_to_numpy(matrix):
    """
    Convert PyTorch tensor to NumPy array if needed
    
    Args:
        matrix: Input matrix (numpy.ndarray or torch.Tensor)
        
    Returns:
        numpy.ndarray: NumPy representation of the matrix
    """
    # Convert to numpy if it's a tensor
    if isinstance(matrix, torch.Tensor):
        # Convert tensor to numpy array
        if matrix.is_cuda:
            matrix = matrix.cpu()
        matrix = matrix.detach().numpy()
    elif isinstance(matrix, np.ndarray):
        matrix = matrix.copy()  # Create a copy to avoid modifying the original
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor")
    
    # Squeeze out dimensions of size 1
    matrix = np.squeeze(matrix)
    
    # Check if the result is a 2D matrix
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D after removing dimensions of size 1, but got shape {matrix.shape} with {matrix.ndim} dimensions")
    
    return matrix


def visualize_matrix_interactive(matrix, title="Matrix Visualization", 
                                cmap="viridis", figsize=(10, 8), 
                                annotate=None, font_size=8):
    """
    Interactive visualization of a matrix with adjustable colormap range
    
    Args:
        matrix: Input matrix (numpy.ndarray or torch.Tensor)
        title (str, optional): Plot title. Defaults to "Matrix Visualization".
        cmap (str, optional): Colormap name. Defaults to "viridis".
        figsize (tuple, optional): Figure size. Defaults to (10, 8).
        annotate (bool or int, optional): Whether to annotate cells with values.
            If None, auto-detect based on matrix size.
            If int, annotate cells if matrix dimensions are <= annotate.
        font_size (int, optional): Font size for annotations. Defaults to 8.
    """
    # Convert to numpy if needed
    matrix_np = convert_to_numpy(matrix)
    
    # Create the figure and the axes
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)
    
    # Auto-detect annotation based on matrix size if not specified
    if annotate is None:
        max_dim = max(matrix_np.shape)
        annotate = max_dim <= 50  # Auto-annotate if matrix is small enough
    elif isinstance(annotate, int):
        annotate = max(matrix_np.shape) <= annotate
    
    # Get min and max values for colorbar
    vmin, vmax = np.min(matrix_np), np.max(matrix_np)
    initial_vmin = vmin
    initial_vmax = vmax
    
    # Range padding for sliders (add 20% margin)
    value_range = vmax - vmin
    slider_vmin = vmin - 0.2 * value_range
    slider_vmax = vmax + 0.2 * value_range
    
    # Initial heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    im = ax.imshow(matrix_np, cmap=cmap, vmin=initial_vmin, vmax=initial_vmax)
    cbar = plt.colorbar(im, cax=cax)
    
    # Annotate cells with values if requested
    texts = []
    if annotate:
        for i in range(matrix_np.shape[0]):
            for j in range(matrix_np.shape[1]):
                # Format value based on magnitude
                value = matrix_np[i, j]
                if abs(value) < 0.01 and value != 0:
                    text = f"{value:.2e}"
                else:
                    text = f"{value:.2f}"
                # Add text annotation to the cell
                txt = ax.text(j, i, text, ha="center", va="center", 
                            color="black" if 0.3 < im.norm(value) < 0.7 else "white",
                            fontsize=font_size)
                texts.append(txt)
    
    ax.set_title(title)
    
    # Add x and y labels with indices
    ax.set_xticks(np.arange(matrix_np.shape[1]))
    ax.set_yticks(np.arange(matrix_np.shape[0]))
    
    # Set axes labels
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    
    # Add sliders for min and max values
    ax_vmin = plt.axes([0.25, 0.1, 0.55, 0.03])
    ax_vmax = plt.axes([0.25, 0.15, 0.55, 0.03])
    
    s_vmin = Slider(ax_vmin, 'Min', slider_vmin, slider_vmax, valinit=initial_vmin)
    s_vmax = Slider(ax_vmax, 'Max', slider_vmin, slider_vmax, valinit=initial_vmax)
    
    # Add buttons for colormap selection
    ax_cmap = plt.axes([0.8, 0.025, 0.15, 0.15])
    popular_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                    'Blues', 'Reds', 'Greens', 'coolwarm', 'RdBu']
    
    # Create radio buttons for colormap selection
    from matplotlib.widgets import RadioButtons
    radio = RadioButtons(ax_cmap, popular_cmaps, active=popular_cmaps.index(cmap))
    
    # Add reset button
    ax_reset = plt.axes([0.05, 0.025, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    # Define update functions
    def update(val):
        vmin = s_vmin.val
        vmax = s_vmax.val
        
        # Ensure vmin < vmax
        if vmin >= vmax:
            vmin = vmax - 0.01
            s_vmin.set_val(vmin)
        
        im.set_clim(vmin, vmax)
        
        # Update text colors based on new normalization
        if annotate:
            for i in range(matrix_np.shape[0]):
                for j in range(matrix_np.shape[1]):
                    value = matrix_np[i, j]
                    texts[i * matrix_np.shape[1] + j].set_color(
                        "black" if 0.3 < im.norm(value) < 0.7 else "white")
        
        fig.canvas.draw_idle()
    
    def update_cmap(val):
        im.set_cmap(val)
        fig.canvas.draw_idle()
    
    def reset(event):
        s_vmin.set_val(initial_vmin)
        s_vmax.set_val(initial_vmax)
        radio.set_active(popular_cmaps.index(cmap))
        im.set_cmap(cmap)
        fig.canvas.draw_idle()
    
    # Connect the update functions
    s_vmin.on_changed(update)
    s_vmax.on_changed(update)
    radio.on_clicked(update_cmap)
    button_reset.on_clicked(reset)
    
    plt.tight_layout()
    plt.show()


def save_matrix_heatmap(matrix, filename="matrix_heatmap.png", title="Matrix Heatmap", 
                         cmap="viridis", figsize=(10, 8), dpi=300, 
                         annotate=None, font_size=8):
    """
    Save a matrix visualization as a heatmap PNG file
    
    Args:
        matrix: Input matrix (numpy.ndarray or torch.Tensor)
        filename (str, optional): Output file path. Defaults to "matrix_heatmap.png".
        title (str, optional): Plot title. Defaults to "Matrix Heatmap".
        cmap (str, optional): Colormap name. Defaults to "viridis".
        figsize (tuple, optional): Figure size. Defaults to (10, 8).
        dpi (int, optional): Image resolution. Defaults to 300.
        annotate (bool or int, optional): Whether to annotate cells with values.
            If None, auto-detect based on matrix size.
            If int, annotate cells if matrix dimensions are <= annotate.
        font_size (int, optional): Font size for annotations. Defaults to 8.
    """
    # Convert to numpy if needed
    matrix_np = convert_to_numpy(matrix)
    
    # Auto-detect annotation based on matrix size if not specified
    if annotate is None:
        max_dim = max(matrix_np.shape)
        annotate = max_dim <= 50  # Auto-annotate if matrix is small enough
    elif isinstance(annotate, int):
        annotate = max(matrix_np.shape) <= annotate
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn for a nicer heatmap
    sns.heatmap(matrix_np, ax=ax, cmap=cmap, annot=annotate, fmt='.2g' if annotate else None,
                annot_kws={'size': font_size} if annotate else None)
    
    ax.set_title(title)
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)) if os.path.dirname(filename) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"Heatmap saved to {os.path.abspath(filename)}")


# Example usage (will run if script is executed directly)
if __name__ == "__main__":
    # Create a sample matrix
    sample_matrix = np.random.randn(10, 10)
    
    # Interactive visualization
    print("Displaying interactive visualization...")
    visualize_matrix_interactive(sample_matrix, title="Random Matrix Demo", annotate=True)
    
    # Save heatmap
    print("Saving heatmap to 'sample_heatmap.png'...")
    save_matrix_heatmap(sample_matrix, "sample_heatmap.png", title="Random Matrix Heatmap", annotate=True)