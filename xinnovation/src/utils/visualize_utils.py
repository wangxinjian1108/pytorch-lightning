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
from xinnovation.src.core.dataclass import TrajParamIndex
from matplotlib.patches import Rectangle, Polygon
import matplotlib.transforms as transforms
from enum import IntEnum
import os


__all__ = [
    'convert_to_numpy',
    'visualize_matched_trajs_on_bev',
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

    
def create_polygon_from_traj(x, y, length, width, yaw, edgecolor, facecolor, alpha, linewidth):
    # 局部坐标下的四个角点（逆时针）
    half_l, half_w = length / 2, width / 2
    corners = np.array([
        [-half_l, -half_w],
        [ half_l, -half_w],
        [ half_l,  half_w],
        [-half_l,  half_w]
    ])
    
    # rotate by yaw
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    
    rotated_corners = corners @ rotation_matrix.T

    # 平移到中心
    translated_corners = rotated_corners + np.array([x, y])
    
    # inverse 0 and 1 column
    regularized_corners = np.hstack([translated_corners[:, 1:2], translated_corners[:, 0:1]])
    # NOTE: 需要颠倒xy，适应ax的坐标系方向
    
    return Polygon(
        regularized_corners,
        closed=True,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        linewidth=linewidth
    )

def visualize_matched_trajs_on_bev(gt_trajs, pred_trajs, gt_idx, pred_idx, save_path, range=[-100,100,-12,12]):
    """
    Visualize ground truth and predicted trajectories on BEV with matching information.
    
    Args:
        gt_trajs: Ground truth trajectories [M, TrajParamIndex.END_OF_INDEX]
        pred_trajs: Predicted trajectories [N, TrajParamIndex.END_OF_INDEX]
        gt_idx: List of matched ground truth indices
        pred_idx: List of matched prediction indices (corresponding to gt_idx)
        save_path: Path to save the visualization
    """
    # convert to numpy array
    if isinstance(gt_trajs, torch.Tensor):
        gt_trajs = gt_trajs.detach().cpu().numpy()
    
    if isinstance(pred_trajs, torch.Tensor):
        pred_trajs = pred_trajs.detach().cpu().numpy()
        
    # Define colors
    gt_color = 'green'
    pred_color = 'blue'
    unmatched_pred_color = 'black'
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 40))
    
    # Set the BEV range limits
    x_min, x_max = range[0], range[1]
    y_min, y_max = range[2], range[3]
    
    # Set axis limits for BEV display
    ax.set_xlim(y_max, y_min)  # Note: in BEV, x-axis typically represents y in vehicle coordinates
    ax.set_ylim(x_min, x_max)  # and y-axis represents x in vehicle coordinates
    
    # Define transparency levels
    gt_alpha = 0.7
    pred_alpha = 0.5
    unmatched_alpha = 0.3
    
    # Create sets for O(1) lookup
    matched_gt_indices = set(gt_idx)
    matched_pred_indices = set(pred_idx)
    
    # Create a mapping from gt_idx to pred_idx for quick lookup
    gt_to_pred_map = {gt: pred for gt, pred in zip(gt_idx, pred_idx)}
    
    # Draw matched ground truth trajectories
    for i, traj in enumerate(gt_trajs):
        if i in matched_gt_indices:
            # Extract relevant parameters
            x = traj[TrajParamIndex.X]
            y = traj[TrajParamIndex.Y]
            length = traj[TrajParamIndex.LENGTH]
            width = traj[TrajParamIndex.WIDTH]
            
            # Calculate yaw angle from cos and sin
            cos_yaw = traj[TrajParamIndex.COS_YAW]
            sin_yaw = traj[TrajParamIndex.SIN_YAW]
            yaw = np.arctan2(sin_yaw, cos_yaw)
            
            # if traj[TrajParamIndex.VX] > 0.1:
            # yaw = np.arctan2(TrajParamIndex.VY, TrajParamIndex.VX)
            
            poly = create_polygon_from_traj(x, y, length, width, yaw, gt_color, gt_color, gt_alpha, 1.5)
            ax.add_patch(poly)
            
            # Add ID text to the center of the box
            ax.text(y, x, f"GT{i}", color='white', fontweight='bold', ha='center', va='center')
            
            # Connect with matched prediction using a line
            matched_pred_id = gt_to_pred_map[i]
            matched_pred = pred_trajs[matched_pred_id]
            pred_x = matched_pred[TrajParamIndex.X]
            pred_y = matched_pred[TrajParamIndex.Y]
            ax.plot([y, pred_y], [x, pred_x], 'r--', alpha=0.5)
    
    # Draw all prediction trajectories
    for i, traj in enumerate(pred_trajs):
        # Extract relevant parameters
        x = traj[TrajParamIndex.X]
        y = traj[TrajParamIndex.Y]
        length = traj[TrajParamIndex.LENGTH]
        width = traj[TrajParamIndex.WIDTH]
        
        # Calculate yaw angle from cos and sin
        cos_yaw = traj[TrajParamIndex.COS_YAW]
        sin_yaw = traj[TrajParamIndex.SIN_YAW]
        yaw = np.arctan2(sin_yaw, cos_yaw)
        
        # if traj[TrajParamIndex.VX] > 0.1:
        #     yaw = np.arctan2(TrajParamIndex.VY, TrajParamIndex.VX)
        
        # Determine color based on matching status
        color = pred_color if i in matched_pred_indices else unmatched_pred_color
        alpha = pred_alpha if i in matched_pred_indices else unmatched_alpha
        
        poly = create_polygon_from_traj(x, y, length, width, yaw, color, color, alpha, 1.5)
        ax.add_patch(poly)
        
        # Add ID text to the center of the box
        text_color = 'lightblue' if i in matched_pred_indices else 'lightgray'
        ax.text(y, x, f"{i}", color=text_color, fontweight='bold', ha='center', va='center')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set title and labels
    ax.set_title(f"BEV Trajectory Visualization\nMatched pairs: {len(gt_idx)}/{len(gt_trajs)} GT, {len(pred_idx)}/{len(pred_trajs)} Pred")
    ax.set_xlabel("Y (meters)")
    ax.set_ylabel("X (meters)")
    
    # 画点并设置图例标签
    # ax.scatter([5, 10], [10, 15], color='blue', s=50, marker='o', label='Matched Prediction')
    # ax.scatter([-5, -10], [10, 20], color='gray', s=50, marker='x', label='Unmatched Prediction')
    # ax.scatter([3, 6], [5, 12], color='green', s=50, marker='s', label='Ground Truth')
    
    legend_boxes = [
        Rectangle((0, 0), 1, 1, edgecolor=gt_color, facecolor=gt_color, label='Ground Truth'),
        Rectangle((0, 0), 1, 1, edgecolor=pred_color, facecolor=pred_color, label='Matched Prediction'),
        Rectangle((0, 0), 1, 1, edgecolor=unmatched_pred_color, facecolor=unmatched_pred_color, label='Unmatched Prediction'),
    ]
    
    # 添加 legend 到 ax
    ax.legend(handles=legend_boxes, loc='upper right')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=gt_color, edgecolor=gt_color, alpha=gt_alpha, label='Matched GT'),
        Patch(facecolor=pred_color, edgecolor=pred_color, alpha=pred_alpha, label='Matched Pred'),
        Patch(facecolor=unmatched_pred_color, edgecolor=unmatched_pred_color, alpha=unmatched_alpha, label='Unmatched Pred')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Make sure directory exists
    # os.makedirs(os.path.dirname(os.path.abspath(save_path)) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"BEV visualization saved to: {os.path.abspath(save_path)}")
    

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