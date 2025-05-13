import torch
import math
import numpy as np
import cv2
from xinnovation.src.core import TrajParamIndex

def quaternion2RotationMatix(qw, qx, qy, qz) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix
    :param qw: w part of quaternion
    :param qx: x part of quaternion
    :param qy: y part of quaternion
    :param qz: z part of quaternion
    :return: rotation matrix
    """
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw = qw / norm
    qx = qx / norm
    qy = qy / norm
    qz = qz / norm
    
    return torch.tensor([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

def generate_unit_cube_points(num_points: int = 25):
    """Generate sample points on faces of unit cube.
    
    Args:
        num_points: Number of points to sample on each face
        
    Returns:
        Tensor of shape [3, num_points*6 + 1] containing sampled points
    """
    points = []
    points_per_face = int(np.sqrt(num_points))  # e.g., 5 for 25 points per face
    
    # Sample points on each face
    for dim in range(3):  # x, y, z
        for sign in [-1, 1]:  # negative and positive faces
            # Create grid on face
            if dim == 0:  # yz plane
                y = torch.linspace(-1, 1, points_per_face)
                z = torch.linspace(-1, 1, points_per_face)
                grid_y, grid_z = torch.meshgrid(y, z, indexing='ij')
                x = torch.full_like(grid_y, sign)
                points.append(torch.stack([x, grid_y, grid_z], dim=-1))
                
            elif dim == 1:  # xz plane
                x = torch.linspace(-1, 1, points_per_face)
                z = torch.linspace(-1, 1, points_per_face)
                grid_x, grid_z = torch.meshgrid(x, z, indexing='ij')
                y = torch.full_like(grid_x, sign)
                points.append(torch.stack([grid_x, y, grid_z], dim=-1))
                
            else:  # xy plane
                x = torch.linspace(-1, 1, points_per_face)
                y = torch.linspace(-1, 1, points_per_face)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                z = torch.full_like(grid_x, sign)
                points.append(torch.stack([grid_x, grid_y, z], dim=-1))
    
    points = torch.cat([p.reshape(-1, 3) for p in points], dim=0)
    # add origin point
    points = torch.cat([torch.zeros(1, 3), points], dim=0)
    points = points.transpose(0, 1) # [3, P]
    points /= 2.0
    return points


def generate_bbox_corners_points(with_origin: bool=True) -> torch.Tensor:
    """Generate sample points on corners of 3D bounding box.
    
    Returns:
        Tensor of shape [3, P] containing sampled points
    """
    corners = torch.tensor([
        [0.5, 0.5, -0.5], # front left bottom
        [0.5, -0.5, -0.5], # front right bottom
        [-0.5, -0.5, -0.5], # rear right bottom
        [-0.5, 0.5, -0.5], # rear left bottom
        [0.5, 0.5, 0.5], # front left top
        [0.5, -0.5, 0.5], # front right top
        [-0.5, -0.5, 0.5], # rear right top
        [-0.5, 0.5, 0.5], # rear left top
    ]).transpose(0, 1) # [3, 8]
    if with_origin:
        corners = torch.cat([torch.zeros(3, 1), corners], dim=1)
    return corners

def sample_bbox_edge_points(n_points_per_edge: int, include_corners: bool=True) -> torch.Tensor:
    """
    Sample points uniformly along each edge of a 3D bounding box.
    
    Args:
        n_points_per_edge: Number of points to sample on each edge (excluding corners)
        include_corners: Whether to include corner points in the output
        
    Returns:
        Tensor containing sampled points on edges of the bounding box
        shape: [3, P]
    """
    # Define the 8 corners of the bounding box
    corners = torch.tensor([
        [0.5, 0.5, -0.5],   # 0: front left bottom
        [0.5, -0.5, -0.5],  # 1: front right bottom
        [-0.5, -0.5, -0.5], # 2: rear right bottom
        [-0.5, 0.5, -0.5],  # 3: rear left bottom
        [0.5, 0.5, 0.5],    # 4: front left top
        [0.5, -0.5, 0.5],   # 5: front right top
        [-0.5, -0.5, 0.5],  # 6: rear right top
        [-0.5, 0.5, 0.5],   # 7: rear left top
    ])
    
    # Define the 12 edges by pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
    ]
    
    # List to collect all points
    all_points = []
    
    # Handle corner points if requested
    if include_corners:
        all_points.append(corners)
    
    # Generate points along each edge
    for i, j in edges:
        # Get the start and end points of this edge
        start, end = corners[i], corners[j]
        
        # Create evenly spaced points along the edge (excluding endpoints)
        if n_points_per_edge > 0:
            t = torch.linspace(0, 1, n_points_per_edge + 2)[1:-1]
            
            # Linear interpolation
            points = start + t.unsqueeze(1) * (end - start)
            all_points.append(points)
    
    points = torch.cat(all_points, dim=0).transpose(0, 1)
    
    # Combine all points into a single tensor (3, P)
    return points

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse the sigmoid function."""
    x_safe = x.clamp(min=eps, max=1 - eps)
    return torch.log(x_safe / (1 - x_safe))


def conditional_amin(x, invalid_mask):
    """Compute amin based on mask, if all elements are masked, return 0, keeping the dimensions."""
    # Mask invalid elements with inf so they won't contribute to the min
    masked_x = x.masked_fill(invalid_mask, float('inf'))
    
    # Compute the min along the last dimension
    amin, _ = masked_x.min(dim=-1, keepdim=True)
    
    # Check if all elements are masked (i.e., if all values are inf)
    all_masked = invalid_mask.all(dim=-1, keepdim=True)
    
    # If all elements are masked, replace the result with zeros (keeping the dimension)
    amin = torch.where(all_masked, torch.zeros_like(amin), amin)
    
    return amin


def conditional_amax(x, invalid_mask):
    """Compute amax based on mask, if all elements are masked, return 0, keeping the dimensions."""
    # Mask invalid elements with -inf so they won't contribute to the max
    masked_x = x.masked_fill(invalid_mask, float('-inf'))
    
    # Compute the max along the last dimension
    amax, _ = masked_x.max(dim=-1, keepdim=True)
    
    # Check if all elements are masked (i.e., if all values are -inf)
    all_masked = invalid_mask.all(dim=-1, keepdim=True)
    
    # If all elements are masked, replace the result with zeros (keeping the dimension)
    amax = torch.where(all_masked, torch.zeros_like(amax), amax)
    
    return amax


def generate_bbox2D_from_pixel_cloud(pixel_clouds, center_format: bool = False, drop_neg_elements: bool = True) -> torch.Tensor:
    """Generate 2D bounding box from pixel cloud.
    
    Args:
        pixel_clouds: Tensor of shape [..., P, 2]
        center_format: Whether to return bounding box in center format (x_center, y_center, width, height).
        drop_neg_elements: If True, ignore negative elements in the pixel cloud when computing bbox.
        
    Returns:
        Tensor of shape [..., 4] containing 2D bounding box.
    """
    if drop_neg_elements:
        # Clone to avoid modifying the original tensor
        x_coords = pixel_clouds[..., 0].clone()  # shape [..., P]
        y_coords = pixel_clouds[..., 1].clone()  # shape [..., P]
        
        # Create mask for valid (non-negative) elements
        invalid_mask = (x_coords < 0) | (y_coords < 0)
        
        min_x = conditional_amin(x_coords, invalid_mask)
        max_x = conditional_amax(x_coords, invalid_mask)
        min_y = conditional_amin(y_coords, invalid_mask)
        max_y = conditional_amax(y_coords, invalid_mask)
        
    else:
        # Compute min and max directly
        x_coords = pixel_clouds[..., :, 0]  # shape [..., P]
        y_coords = pixel_clouds[..., :, 1]  # shape [..., P]
        min_x, _ = x_coords.min(dim=-1)
        max_x, _ = x_coords.max(dim=-1)
        min_y, _ = y_coords.min(dim=-1)
        max_y, _ = y_coords.max(dim=-1)
    
    min_x = min_x.squeeze(-1)
    min_y = min_y.squeeze(-1)
    max_x = max_x.squeeze(-1)
    max_y = max_y.squeeze(-1)
    # Create the bounding box
    if center_format:
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        valid_mask = (width > 0) & (height > 0)
        bbox_2d = torch.stack([center_x, center_y, width, height, valid_mask], dim=-1)
    else:
        valid_mask = torch.ones_like(pixel_clouds[..., 0, 0])
        bbox_2d = torch.stack([min_x, min_y, max_x, max_y, valid_mask], dim=-1)
    
    return bbox_2d


def combine_multiple_images(img_list, output_shape=None):
    """Combine multiple images into one.
    
    Args:
        img_list: List of images, each image is a numpy array
        output_shape: Shape of the output image, (height, width)
        
    Returns:
        Combined image
    """
    if output_shape is None:
        # take the min shape of img_list
        output_shape = np.min([img.shape[:2] for img in img_list], axis=0) #(H, W)
    img_list = [cv2.resize(img, (output_shape[1], output_shape[0])) for img in img_list]
    row_nb = int(np.ceil(np.sqrt(len(img_list))))
    col_nb = (len(img_list) + row_nb - 1) // row_nb
    combined_img = np.zeros((row_nb * output_shape[0], col_nb * output_shape[1], 3), dtype=np.uint8)
    for i, img in enumerate(img_list):
        row_idx = i // col_nb
        col_idx = i % col_nb
        combined_img[row_idx * output_shape[0]:(row_idx + 1) * output_shape[0],
                     col_idx * output_shape[1]:(col_idx + 1) * output_shape[1]] = img
    return combined_img