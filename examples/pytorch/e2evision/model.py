import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple
import torch.nn.functional as F
from base import (
    SourceCameraId, CameraType, ObstacleTrajectory, 
    ObjectType, TrajParamIndex, Point3DAccMotion, AttributeType
)
import numpy as np
from temporal_fusion import TemporalAttentionFusion

class ImageFeatureExtractor(nn.Module):
    """2D CNN backbone for extracting image features."""
    def __init__(self, out_channels: int = 256):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2],  # Remove avg pool and fc
            nn.Conv2d(2048, out_channels, 1),  # Reduce channel dimension
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    

class TrajectoryDecoder(nn.Module):
    """Trajectory decoder with iterative refinement and feature sampling."""
    def __init__(self,
                 num_layers: int = 6,
                 num_queries: int = 100,
                 feature_dim: int = 256,
                 hidden_dim: int = 512,
                 num_points: int = 24):  # Points to sample on 3D box (4 per face)
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_points = num_points
        
        # Learnable trajectory queries
        self.trajectory_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # 1. Motion parameter head - predict normalized 0-1 values for motion & size
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, TrajParamIndex.END_OF_INDEX),
            nn.Sigmoid()  # Output normalized 0-1 values
        )
        
        # 2. Object type classification head
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ObjectType))
        )
        
        # 3. Attribute prediction head
        self.attribute_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, AttributeType.END_OF_INDEX)
        )
        
        # Parameter scaling values (learnable)
        self.param_ranges = nn.ParameterDict({
            'min': nn.Parameter(torch.tensor([
                -50.0, -50.0, -5.0,        # x, y, z position
                -15.0, -15.0,              # vx, vy velocity
                -5.0, -5.0,                # ax, ay acceleration
                -3.14,                     # yaw
                0.5, 0.5, 0.5              # length, width, height
            ])),
            'max': nn.Parameter(torch.tensor([
                50.0, 50.0, 5.0,           # x, y, z position
                15.0, 15.0,                # vx, vy velocity
                5.0, 5.0,                  # ax, ay acceleration
                3.14,                      # yaw
                10.0, 5.0, 3.0             # length, width, height
            ])),
        })
        
        # Feature aggregation MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Query update transformer layers
        self.layers = nn.ModuleList([
            TrajectoryRefinementLayer(
                hidden_dim=hidden_dim,
                num_heads=8
            ) for _ in range(num_layers)
        ])
    
    def predict_trajectory_parameters(self, queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict trajectory parameters from query embeddings.
        
        Args:
            queries: Tensor[B, N, hidden_dim]
            
        Returns:
            Dict containing:
                - 'motion_params': Tensor[B, N, TrajParamIndex.END_OF_INDEX]
                - 'object_type': Tensor[B, N] - Class indices
                - 'attributes': Tensor[B, N, AttributeType.END_OF_INDEX]
        """
        B, N = queries.shape[:2]
        
        # 1. Predict motion parameters (normalized 0-1)
        norm_params = self.motion_head(queries)  # [B, N, TrajParamIndex.END_OF_INDEX]
        
        # Scale parameters to real values
        min_vals = self.param_ranges['min']
        max_vals = self.param_ranges['max']
        motion_params = norm_params * (max_vals - min_vals) + min_vals
        
        # 2. Predict object type
        type_logits = self.type_head(queries)  # [B, N, len(ObjectType)]
        type_probs = F.softmax(type_logits, dim=-1)
        object_type = torch.argmax(type_probs, dim=-1)  # [B, N]
        
        # 3. Predict attributes
        attribute_logits = self.attribute_head(queries)  # [B, N, AttributeType.END_OF_INDEX]
        attributes = torch.sigmoid(attribute_logits)  # Each attribute is binary
        
        return {
            'motion_params': motion_params,
            'object_type': object_type,
            'attributes': attributes,
            'type_logits': type_logits
        }
    
    def forward(self, 
                features_dict: Dict[SourceCameraId, torch.Tensor],
                calibrations: Dict[SourceCameraId, Dict],
                ego_states: torch.Tensor) -> Tuple[List[ObstacleTrajectory], List[Dict]]:
        """
        Args:
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> calibration_dict]
            ego_states: Tensor[B, T, 5] containing position, yaw, timestamp
            
        Returns:
            - Final trajectories list
            - List of intermediate prediction dicts for auxiliary losses
        """
        B = next(iter(features_dict.values())).shape[0]
        T = next(iter(features_dict.values())).shape[1]
        queries = self.trajectory_queries.expand(B, -1, -1)
        
        all_predictions = []
        
        # Iterative refinement
        for layer_idx, layer in enumerate(self.layers):
            # 1. Predict trajectory parameters from queries
            predictions = self.predict_trajectory_parameters(queries) # [B, N, TrajParamIndex.END_OF_INDEX]
            all_predictions.append(predictions)
            
            # For the last layer, we don't need to update queries
            if layer_idx == self.num_layers - 1:
                break
                
            # 2. Sample points on 3D bounding boxes using motion parameters
            box_points = self.sample_box_points(predictions['motion_params'])  # [B, N, num_points, 3]
            
            # 3. Gather features from all views and frames
            point_features = self.gather_point_features(
                box_points, features_dict, calibrations, ego_states
            )  # [B, N, num_points, T, num_cameras, C]
            
            # 4. Aggregate features
            agg_features = self.aggregate_features(point_features)  # [B, N, hidden_dim]
            
            # 5. Update queries
            queries = layer(queries, agg_features)
        
        # Convert final parameters to ObstacleTrajectory objects
        final_predictions = all_predictions[-1]
        trajectories = self.params_to_trajectories(
            final_predictions['motion_params'][0],  # Use first batch
            final_predictions['object_type'][0],
            final_predictions['attributes'][0]
        )
        
        return trajectories, all_predictions
    
    def params_to_trajectories(self, 
                             motion_params: torch.Tensor, 
                             object_types: torch.Tensor,
                             attributes: torch.Tensor) -> List[ObstacleTrajectory]:
        """Convert predicted parameters to ObstacleTrajectory objects.
        
        Args:
            motion_params: Tensor[N, TrajParamIndex.END_OF_INDEX]
            object_types: Tensor[N] - Object type indices
            attributes: Tensor[N, AttributeType.END_OF_INDEX] - Attribute probabilities
            
        Returns:
            List[ObstacleTrajectory]
        """
        trajectories = []
        
        # Extract has_object probabilities
        has_object_probs = attributes[:, AttributeType.HAS_OBJECT]
        is_static_probs = attributes[:, AttributeType.STATIC]
        is_occluded_probs = attributes[:, AttributeType.OCCLUDED]
        
        # Filter out detections with low confidence
        valid_mask = has_object_probs > 0.5
        
        for i in range(motion_params.shape[0]):
            if not valid_mask[i]:
                continue
                
            p = motion_params[i]
            
            # Create Point3DAccMotion
            motion = Point3DAccMotion(
                x=p[TrajParamIndex.X].item(),
                y=p[TrajParamIndex.Y].item(),
                z=p[TrajParamIndex.Z].item(),
                vx=p[TrajParamIndex.VX].item(),
                vy=p[TrajParamIndex.VY].item(),
                vz=0.0,
                ax=p[TrajParamIndex.AX].item(),
                ay=p[TrajParamIndex.AY].item(),
                az=0.0
            )
            
            # Get static flag from attributes
            is_static = bool(is_static_probs[i].item() > 0.5)
            
            # Create ObstacleTrajectory
            traj = ObstacleTrajectory(
                id=i,
                motion=motion,
                yaw=p[TrajParamIndex.YAW].item(),
                length=p[TrajParamIndex.LENGTH].item(),
                width=p[TrajParamIndex.WIDTH].item(),
                height=p[TrajParamIndex.HEIGHT].item(),
                object_type=ObjectType(object_types[i].item()),
                static=is_static,
                valid=True
            )
            
            trajectories.append(traj)
        
        return trajectories
    
    def sample_box_points(self, traj_params: torch.Tensor) -> torch.Tensor:
        """Sample points on 3D bounding box faces.
        
        Args:
            traj_params: Tensor[B, N, num_params]
            
        Returns:
            Tensor[B, N, num_points, 3] - 3D points
        """
        B, N = traj_params.shape[:2]
        device = traj_params.device
        
        # Extract box dimensions
        center_x = traj_params[..., TrajParamIndex.X]
        center_y = traj_params[..., TrajParamIndex.Y]
        center_z = traj_params[..., TrajParamIndex.Z]
        length = traj_params[..., TrajParamIndex.LENGTH]
        width = traj_params[..., TrajParamIndex.WIDTH]
        height = traj_params[..., TrajParamIndex.HEIGHT]
        yaw = traj_params[..., TrajParamIndex.YAW]
        
        # Create sampling grid for each face (4 points per face, 6 faces)
        points_per_face = self.num_points // 6
        
        # Sample relative coordinates (-0.5 to 0.5)
        offsets = torch.linspace(-0.4, 0.4, points_per_face, device=device)
        
        # Generate points for each face
        all_points = []
        
        # Front face (x = -l/2)
        for i in range(points_per_face):
            for j in range(points_per_face):
                point = torch.stack([
                    torch.full_like(center_x, -0.5),  # x = -l/2
                    offsets[i].expand_like(center_y),  # y varies
                    offsets[j].expand_like(center_z)   # z varies
                ], dim=-1)
                all_points.append(point)
        
        # Back face (x = l/2)
        for i in range(points_per_face):
            for j in range(points_per_face):
                point = torch.stack([
                    torch.full_like(center_x, 0.5),   # x = l/2
                    offsets[i].expand_like(center_y),  # y varies
                    offsets[j].expand_like(center_z)   # z varies
                ], dim=-1)
                all_points.append(point)
        
        # Left face (y = -w/2)
        for i in range(points_per_face):
            for j in range(points_per_face):
                point = torch.stack([
                    offsets[i].expand_like(center_x),  # x varies
                    torch.full_like(center_y, -0.5),   # y = -w/2
                    offsets[j].expand_like(center_z)   # z varies
                ], dim=-1)
                all_points.append(point)
        
        # Right face (y = w/2)
        for i in range(points_per_face):
            for j in range(points_per_face):
                point = torch.stack([
                    offsets[i].expand_like(center_x),  # x varies
                    torch.full_like(center_y, 0.5),    # y = w/2
                    offsets[j].expand_like(center_z)   # z varies
                ], dim=-1)
                all_points.append(point)
        
        # Bottom face (z = -h/2)
        for i in range(points_per_face):
            for j in range(points_per_face):
                point = torch.stack([
                    offsets[i].expand_like(center_x),  # x varies
                    offsets[j].expand_like(center_y),  # y varies
                    torch.full_like(center_z, -0.5)    # z = -h/2
                ], dim=-1)
                all_points.append(point)
        
        # Top face (z = h/2)
        for i in range(points_per_face):
            for j in range(points_per_face):
                point = torch.stack([
                    offsets[i].expand_like(center_x),  # x varies
                    offsets[j].expand_like(center_y),  # y varies
                    torch.full_like(center_z, 0.5)     # z = h/2
                ], dim=-1)
                all_points.append(point)
        
        # Stack all points
        box_points = torch.stack(all_points, dim=2)  # [B, N, num_points, 3]
        
        # Scale by box dimensions
        box_points[..., 0] *= length.unsqueeze(-1)
        box_points[..., 1] *= width.unsqueeze(-1)
        box_points[..., 2] *= height.unsqueeze(-1)
        
        # Apply rotation
        cos_yaw = torch.cos(yaw).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).unsqueeze(-1)
        
        x_rotated = box_points[..., 0] * cos_yaw - box_points[..., 1] * sin_yaw
        y_rotated = box_points[..., 0] * sin_yaw + box_points[..., 1] * cos_yaw
        
        box_points[..., 0] = x_rotated
        box_points[..., 1] = y_rotated
        
        # Translate to center
        box_points[..., 0] += center_x.unsqueeze(-1)
        box_points[..., 1] += center_y.unsqueeze(-1)
        box_points[..., 2] += center_z.unsqueeze(-1)
        
        return box_points
    
    def gather_point_features(self,
                            box_points: torch.Tensor,
                            features_dict: Dict[SourceCameraId, torch.Tensor],
                            calibrations: Dict[SourceCameraId, Dict],
                            ego_states: torch.Tensor) -> torch.Tensor:
        """Gather features for 3D points from all cameras and frames.
        
        Args:
            box_points: Tensor[B, N, num_points, 3]
            features_dict: Dict[camera_id -> Tensor[B, T, C, H, W]]
            calibrations: Dict[camera_id -> calibration_dict]
            ego_states: Tensor[B, T, 5]
            
        Returns:
            Tensor[B, N, num_points, T, num_cameras, C]
        """
        B, N, P = box_points.shape[:3]
        T = next(iter(features_dict.values())).shape[1]
        C = next(iter(features_dict.values())).shape[2]
        num_cameras = len(features_dict)
        
        # Initialize output tensor
        point_features = []
        
        # Process each camera view
        for camera_idx, (camera_id, features) in enumerate(features_dict.items()):
            camera_features = []
            
            # For each time step
            for t in range(T):
                # Transform points to ego frame at time t
                t_points = self.transform_points_to_ego_frame(
                    box_points, ego_states[:, t]
                )  # [B, N, P, 3]
                
                # Project to camera view
                points_2d = self.project_points_to_image(
                    t_points.reshape(B*N*P, 3),
                    calibrations[camera_id]['intrinsic'],
                    calibrations[camera_id]['extrinsic']
                )  # [B*N*P, 2]
                
                # Sample features
                H, W = features.shape[3:5]
                
                # Normalize to [-1, 1] for grid_sample
                points_2d[..., 0] = 2 * points_2d[..., 0] / W - 1
                points_2d[..., 1] = 2 * points_2d[..., 1] / H - 1
                
                # Check if points are in image bounds
                valid_mask = (
                    (points_2d[..., 0] >= -1) & (points_2d[..., 0] <= 1) &
                    (points_2d[..., 1] >= -1) & (points_2d[..., 1] <= 1)
                ).float().reshape(B, N, P, 1)
                
                # Clamp values to prevent sampling outside
                points_2d = torch.clamp(points_2d, -1, 1)
                
                # Sample features
                features_t = features[:, t]  # [B, C, H, W]
                sampled_feats = F.grid_sample(
                    features_t,
                    points_2d.reshape(B, -1, 1, 2).to(features_t.device),
                    mode='bilinear',
                    align_corners=False
                )  # [B, C, N*P, 1]
                
                # Reshape and apply valid mask
                sampled_feats = sampled_feats.reshape(B, C, N, P).permute(0, 2, 3, 1)  # [B, N, P, C]
                sampled_feats = sampled_feats * valid_mask
                
                camera_features.append(sampled_feats)
            
            # Stack features for this camera
            camera_features = torch.stack(camera_features, dim=3)  # [B, N, P, T, C]
            point_features.append(camera_features)
        
        # Stack all camera features
        point_features = torch.stack(point_features, dim=4)  # [B, N, P, T, num_cameras, C]
        
        return point_features
    
    def transform_points_to_ego_frame(self, 
                                    points: torch.Tensor,
                                    ego_state: torch.Tensor) -> torch.Tensor:
        """Transform 3D points to ego vehicle frame at given time.
        
        Args:
            points: Tensor[B, N, P, 3]
            ego_state: Tensor[B, 5] with position, yaw, timestamp
            
        Returns:
            Tensor[B, N, P, 3]
        """
        B, N, P = points.shape[:3]
        
        # Extract ego position and yaw
        ego_pos = ego_state[..., :3].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 3]
        ego_yaw = ego_state[..., 3].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        
        # Create rotation matrix
        cos_yaw = torch.cos(ego_yaw)
        sin_yaw = torch.sin(ego_yaw)
        
        # Apply inverse transform (ego->world to world->ego)
        # First translate
        translated = points - ego_pos
        
        # Then rotate (inverse rotation)
        x_rotated = translated[..., 0] * cos_yaw + translated[..., 1] * sin_yaw
        y_rotated = -translated[..., 0] * sin_yaw + translated[..., 1] * cos_yaw
        
        transformed = torch.stack(
            [x_rotated, y_rotated, translated[..., 2]], 
            dim=-1
        )
        
        return transformed
    
    def aggregate_features(self, point_features: torch.Tensor) -> torch.Tensor:
        """Aggregate features from all points, frames and cameras.
        
        Args:
            point_features: Tensor[B, N, P, T, num_cameras, C]
            
        Returns:
            Tensor[B, N, hidden_dim]
        """
        B, N = point_features.shape[:2]
        
        # Reshape for processing
        features_flat = point_features.flatten(2, 4)  # [B, N, P*T*num_cameras, C]
        
        # Max pooling over all points, frames and cameras
        pooled_features, _ = torch.max(features_flat, dim=2)  # [B, N, C]
        
        # Process through MLP
        processed_features = self.feature_mlp(pooled_features)  # [B, N, hidden_dim]
        
        return processed_features


class TrajectoryRefinementLayer(nn.Module):
    """Layer for refining trajectory queries."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feature attention
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Tensor[B, N, hidden_dim]
            features: Tensor[B, N, hidden_dim]
            
        Returns:
            Tensor[B, N, hidden_dim]
        """
        # Self-attention
        q = self.norm1(queries)
        q = q + self.self_attn(q, q, q)[0]
        
        # Feature attention
        q = self.norm2(q)
        q = q + self.feature_attn(q, features, features)[0]
        
        # FFN
        q = self.norm3(q)
        q = q + self.ffn(q)
        
        return q

class E2EPerceptionNet(nn.Module):
    """End-to-end multi-camera 3D perception network."""
    def __init__(self,
                 camera_ids: List[SourceCameraId],
                 feature_dim: int = 256,
                 num_queries: int = 100,
                 num_decoder_layers: int = 6):
        super().__init__()
        self.camera_ids = camera_ids
        
        # Image feature extraction
        self.feature_extractor = ImageFeatureExtractor(feature_dim)
        
        # Per-camera temporal fusion
        self.temporal_fusion = nn.ModuleDict({
            str(camera_id): TemporalAttentionFusion(
                feature_dim=feature_dim,
                num_heads=8,
                dropout=0.1
            ) for camera_id in camera_ids
        })
        
        # Trajectory decoder
        self.trajectory_decoder = TrajectoryDecoder(
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            feature_dim=feature_dim
        )
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass returning trajectories and auxiliary outputs for training.
        
        Args:
            batch: Dict containing:
                - images: Dict[camera_id -> Tensor[B, T, 3, H, W]]
                - ego_states: List[Dict]
                - calibrations: Dict[camera_id -> Dict]
        """
        # Extract features from each camera
        features = {}
        for camera_id in self.camera_ids:
            images = batch['images'][camera_id]  # [B, T, 3, H, W]
            B, T = images.shape[:2]
            
            # Extract features for all frames
            feat = self.feature_extractor(images.flatten(0, 1))  # [B*T, C, H, W]
            feat = feat.view(B, T, *feat.shape[1:])  # [B, T, C, H, W]
            
            # Apply temporal fusion for each camera independently
            fused_feat = self.temporal_fusion[str(camera_id)](feat)  # [B, T, C, H, W]
            
            # Store fused features
            features[camera_id] = fused_feat
        
        # Decode trajectories using fused features from all cameras
        trajectories, aux_trajectories = self.trajectory_decoder(
            features,
            batch['calibrations'],
            batch['ego_states']
        )
        
        return {
            'trajectories': trajectories,
            'aux_trajectories': aux_trajectories
        }