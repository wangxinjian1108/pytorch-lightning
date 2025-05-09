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
from matplotlib.patches import Rectangle, Polygon, Patch
import matplotlib.transforms as transforms
from enum import IntEnum
import os, glob, shutil


__all__ = [
    'generate_video_from_dir',
    'visualize_matched_trajs_on_bev',
    'visualize_refined_trajs_on_bev',
    'visualize_matrix_interactive',
    'save_matrix_heatmap'
]


def generate_video_from_dir(dir, video_path, fps=10, post_fix="png"):
    img_files = glob.glob(os.path.join(dir, f"*.{post_fix}"))
    img_files.sort()
    temp_dir = os.path.join(dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    for i, img_file in enumerate(img_files):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        shutil.copy2(img_file, frame_path)
    ffmpeg_cmd = f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
    os.system(ffmpeg_cmd)
    print(f"Generated video: {video_path}")
    shutil.rmtree(temp_dir)


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

    
def create_polygon_from(x, y, length, width, yaw, edgecolor, facecolor, alpha, linewidth):
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

def create_polygon_from_traj(traj, edgecolor, facecolor, alpha, linewidth):
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
    
    return create_polygon_from(x, y, length, width, yaw, edgecolor, facecolor, alpha, linewidth)

def visualize_matched_trajs_on_bev(gt_trajs, coarse_trajs, refined_trajs, gt_idx, pred_idx, save_path, bev_range=[-100,100,-12,12], infomation=None, pre_match=True):
    """
    Visualize ground truth and predicted trajectories on BEV with matching information.
    
    Args:
        gt_trajs: Ground truth trajectories [M, TrajParamIndex.END_OF_INDEX]
        coarse_trajs: Predicted trajectories [N, TrajParamIndex.END_OF_INDEX]
        refined_trajs: Predicted trajectories [N, TrajParamIndex.END_OF_INDEX]
        gt_idx: List of matched ground truth indices
        pred_idx: List of matched prediction indices (corresponding to gt_idx)
        save_path: Path to save the visualization
    """
    # convert to numpy array
    if isinstance(gt_trajs, torch.Tensor):
        gt_trajs = gt_trajs.detach().cpu().numpy()
    
    if isinstance(coarse_trajs, torch.Tensor):
        coarse_trajs = coarse_trajs.detach().cpu().numpy()
        
    if isinstance(refined_trajs, torch.Tensor):
        refined_trajs = refined_trajs.detach().cpu().numpy()
        
    # Define colors
    gt_color = 'green'
    refined_color = 'blue'
    unmatched_refined_color = 'black'
    coarse_color = 'yellow'
    fp_traj_color = 'red'
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 40))
    
    # Add info
    if infomation is not None:
        ax.text(
            0.01, 0.99,  # x=0.01, y=0.99 表示轴范围的左上角
            infomation,
            transform=ax.transAxes,  # 保证是相对于坐标轴大小，而不是数据坐标
            fontsize=16,           # 字体大小（比默认大）
            fontweight='bold',     # 加粗
            color='red',
            verticalalignment='top',
            horizontalalignment='left'
        )
    
    # Set the BEV bev_range limits
    x_min, x_max = bev_range[0], bev_range[1]
    y_min, y_max = bev_range[2], bev_range[3]
    
    # Set axis limits for BEV display
    ax.set_xlim(y_max, y_min)  # Note: in BEV, x-axis typically represents y in vehicle coordinates
    ax.set_ylim(x_min, x_max)  # and y-axis represents x in vehicle coordinates
    
    # Define transparency levels
    gt_alpha = 0.7
    refined_alpha = 0.5
    unmatched_refined_alpha = 0.3
    
    # Create sets for O(1) lookup
    matched_gt_indices = set(gt_idx)
    matched_pred_indices = set(pred_idx)
    
    # Create a mapping from gt_idx to pred_idx for quick lookup
    gt_to_pred_map = {gt: pred for gt, pred in zip(gt_idx, pred_idx)}
    
    # Draw matched ground truth trajectories
    for i, traj in enumerate(gt_trajs):
        if i in matched_gt_indices:
            poly = create_polygon_from_traj(traj, gt_color, gt_color, gt_alpha, 1.5)
            ax.add_patch(poly)
            
            # Add ID text to the center of the box
            x, y = traj[TrajParamIndex.X], traj[TrajParamIndex.Y]
            ax.text(y, x, f"GT{i}", color='white', fontweight='bold', ha='center', va='center')
            
            # Connect with matched prediction using a line
            matched_pred_id = gt_to_pred_map[i]
            matched_pred = refined_trajs[matched_pred_id]
            if pre_match:
                matched_pred = coarse_trajs[matched_pred_id]
            pred_x = matched_pred[TrajParamIndex.X]
            pred_y = matched_pred[TrajParamIndex.Y]
            ax.plot([y, pred_y], [x, pred_x], 'r--', alpha=0.5)
    
    # Draw all prediction trajectories
    for i, traj in enumerate(refined_trajs):
        # Determine color based on matching status
        color = refined_color if i in matched_pred_indices else unmatched_refined_color
        alpha = refined_alpha if i in matched_pred_indices else unmatched_refined_alpha
        
        poly = create_polygon_from_traj(traj, color, color, alpha, 1.5)
        ax.add_patch(poly)
        
        # Add ID text to the center of the box
        text_color = 'lightgray' if traj[TrajParamIndex.HAS_OBJECT] < 0.5 else fp_traj_color
        if i in matched_pred_indices:
            text_color = 'lightblue'
        x, y = traj[TrajParamIndex.X], traj[TrajParamIndex.Y]
        ax.text(y, x, f"{i}", color=text_color, fontweight='bold', ha='center', va='center')
        
        # Plot coarse trajs if current traj matched to GT
        if i in matched_pred_indices:
            coarse_traj = coarse_trajs[i]
            poly = create_polygon_from_traj(coarse_traj, coarse_color, coarse_color, 0.5, 1)
            ax.add_patch(poly)
            
            # connect the coarse traj to the refined traj
            coarse_x, coarse_y = coarse_traj[TrajParamIndex.X], coarse_traj[TrajParamIndex.Y]
            ax.plot([coarse_y, y], [coarse_x, x], 'c--', alpha=0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set title and labels
    ax.set_title(f"BEV Trajectory Visualization\nMatched pairs: {len(gt_idx)}/{len(gt_trajs)} GT, {len(pred_idx)}/{len(refined_trajs)} Pred")
    ax.set_xlabel("Y (meters)")
    ax.set_ylabel("X (meters)")
    
    legend_boxes = [
        Rectangle((0, 0), 1, 1, edgecolor=gt_color, facecolor=gt_color, label='Ground Truth'),
        Rectangle((0, 0), 1, 1, edgecolor=coarse_color, facecolor=coarse_color, label='Matched Coarse Traj'),
        Rectangle((0, 0), 1, 1, edgecolor=refined_color, facecolor=refined_color, label='Matched Refined Traj'),
        Rectangle((0, 0), 1, 1, edgecolor=unmatched_refined_color, facecolor=unmatched_refined_color, label='Unmatched Refined Traj'),
        Rectangle((0, 0), 1, 1, edgecolor=fp_traj_color, facecolor=fp_traj_color, label='FP Traj(prob>0.5)'),
    ]
    
    ax.legend(handles=legend_boxes, loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"BEV visualization saved to: {os.path.abspath(save_path)}")
    

def visualize_refined_trajs_on_bev(coarse_trajs, refined_trajs, save_path, bev_range=[-100,100,-12,12], infomation=None):
    """
    Visualize the coarse and refined trajs on BEV
    
    Args:
        coarse_trajs: Coarse Trajs before refinemnt [N, TrajParamIndex.END_OF_INDEX]
        refined_trajs: Refined Trajs after refinement [N, TrajParamIndex.END_OF_INDEX]
        save_path: Path to save the visualization
    """
    # convert to numpy array
    if isinstance(coarse_trajs, torch.Tensor):
        coarse_trajs = coarse_trajs.detach().cpu().numpy()
    
    if isinstance(refined_trajs, torch.Tensor):
        refined_trajs = refined_trajs.detach().cpu().numpy()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 40))
    
    # Add info
    if infomation is not None:
        ax.text(
            0.01, 0.99,  # x=0.01, y=0.99 表示轴范围的左上角
            infomation,
            transform=ax.transAxes,  # 保证是相对于坐标轴大小，而不是数据坐标
            fontsize=16,           # 字体大小（比默认大）
            fontweight='bold',     # 加粗
            color='red',
            verticalalignment='top',
            horizontalalignment='left'
        )
    
    # Set the BEV range limits
    x_min, x_max = bev_range[0], bev_range[1]
    y_min, y_max = bev_range[2], bev_range[3]
    
    # Set axis limits for BEV display
    ax.set_xlim(y_max, y_min)  # Note: in BEV, x-axis typically represents y in vehicle coordinates
    ax.set_ylim(x_min, x_max)  # and y-axis represents x in vehicle coordinates
    
    # Define transparency levels
    coarse_color = 'lightgray'
    refined_color = 'green'
    coarse_alpha = 0.7
    refined_alpha = 0.5
    
    # Draw coarse trajectories
    for i in range(len(coarse_trajs)):
        coarse_traj, refined_traj = coarse_trajs[i], refined_trajs[i]
        poly1 = create_polygon_from_traj(coarse_traj, coarse_color, coarse_color, coarse_alpha, 1)
        poly2 = create_polygon_from_traj(refined_traj, refined_color, refined_color, refined_alpha, 1)
        ax.add_patch(poly1)
        ax.add_patch(poly2)
        
        # add id text
        x1, y1 = coarse_traj[TrajParamIndex.X], coarse_traj[TrajParamIndex.Y]
        ax.text(y1, x1, f"{i}", color='white', fontweight='bold', ha='center', va='center')
        
        # connect with a line
        x2, y2 = refined_traj[TrajParamIndex.X], refined_traj[TrajParamIndex.Y]
        ax.plot([y1, y2], [x1, x2], 'r--', alpha=0.5)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set title and labels
    ax.set_title(f"BEV Trajectory Visualization From Coarse Trajs to Refined Trajs")
    ax.set_xlabel("Y (meters)")
    ax.set_ylabel("X (meters)")
    
    legend_boxes = [
        Rectangle((0, 0), 1, 1, edgecolor=coarse_color, facecolor=coarse_color, label='Coarse Traj'),
        Rectangle((0, 0), 1, 1, edgecolor=refined_color, facecolor=refined_color, label='Refined Traj')
    ]
    ax.legend(handles=legend_boxes, loc='upper right')
    
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