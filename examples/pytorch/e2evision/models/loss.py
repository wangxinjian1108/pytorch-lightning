import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from base import TrajParamIndex
from configs.config import LossConfig

def match_trajectories(
    pred_trajs: torch.Tensor,
    gt_trajs: torch.Tensor,
    frames: int = 10,
    dt: float = 0.1,
    iou_method: str = "iou2"
) -> Tuple[List[Tuple[int, int]], List[int], torch.Tensor]:
    """Match predicted trajectories to ground truth using Hungarian algorithm.
    
    Args:
        pred_trajs: Predicted trajectories [B, N, TrajParamIndex.END_OF_INDEX]
        gt_trajs: Ground truth trajectories [B, M, TrajParamIndex.END_OF_INDEX]
        frames: Number of frames to consider for IoU
        dt: Time step between frames
        iou_method: IoU calculation method ("iou" or "iou2")
        
    Returns:
        Tuple containing:
            - List of (pred_idx, gt_idx) pairs for matched trajectories
            - List of pred_idx for unmatched predictions
            - Cost matrix [N, M]
    """
    device = pred_trajs.device
    N, M = len(pred_trajs), len(gt_trajs)
    
    if M == 0:
        return [], list(range(N)), torch.zeros((N, M), device=device)
    
    # Compute cost matrix
    cost_matrix = torch.zeros((N, M), device=device)
    
    for i in range(N):
        for j in range(M):
            # IoU cost
            if iou_method == "iou2":
                iou = calculate_trajectory_bev_iou2(
                    pred_trajs[i:i+1],
                    gt_trajs[j:j+1],
                    torch.arange(frames, device=device) * dt
                )
            else:
                iou = calculate_trajectory_bev_iou(
                    pred_trajs[i:i+1],
                    gt_trajs[j:j+1],
                    torch.arange(frames, device=device) * dt
                )
            
            # Distance cost
            dist_score = calculate_trajectory_distance_score(pred_trajs[i], gt_trajs[j])
            
            score = iou * 0.6 + dist_score * 0.4
            
            if pred_trajs[i, TrajParamIndex.HAS_OBJECT] < 0.5:
                score *= 0.5
            
            obj_index = int(torch.argmax(pred_trajs[i][TrajParamIndex.CAR:]))
            if gt_trajs[j, obj_index] != 1:
                score *= 0.5
            
            # Combined cost
            cost_matrix[i, j] = 1 - score
    
    # Run Hungarian algorithm
    matches = []
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    matches = list(zip(row_ind, col_ind))
    
    # Find unmatched prediction indices
    matched_pred_indices = set(i for i, _ in matches)
    unmatched_pred_indices = [i for i in range(N) if i not in matched_pred_indices]
    
    return matches, unmatched_pred_indices, cost_matrix

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction."""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        
        self.config = config
        self.weight_dict = config.weight_dict
    
    def forward(self, outputs: List[torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass to compute losses."""
        losses = {}
        
        # Process each decoder layer
        num_layers = len(outputs)
        
        # 只在最后一层进行匹配，获取匹配索引
        final_layer_idx = num_layers - 1
        final_pred_trajs = outputs[final_layer_idx]  # [B, N, TrajParamIndex.END_OF_INDEX]
        gt_trajs = targets['trajs']  # [B, M, TrajParamIndex.END_OF_INDEX]
        
        # 计算最后一层的匹配结果
        indices_list = []
        for b in range(final_pred_trajs.shape[0]):
            # 获取当前批次的预测和真实轨迹
            pred_trajs_b = final_pred_trajs[b]  # [N, TrajParamIndex.END_OF_INDEX]
            gt_trajs_b = gt_trajs[b]  # [M, TrajParamIndex.END_OF_INDEX]
            
            # 计算匹配代价矩阵
            cost_matrix = self._compute_matching_cost(pred_trajs_b, gt_trajs_b)
            
            # 使用匈牙利算法进行匹配
            indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            indices_list.append(indices)
        
        # 使用相同的匹配结果计算每一层的损失
        for idx, pred_trajs in enumerate(outputs):
            if idx < final_layer_idx - 2:
                continue
            # 使用最后一层的匹配结果计算当前层的损失
            layer_losses = self._compute_losses_with_indices(
                pred_trajs=pred_trajs,
                gt_trajs=gt_trajs,
                indices_list=indices_list,
                prefix=f'layer_{idx}_'
            )
            
            losses.update(layer_losses)
        
        # 计算总损失
        total_loss = sum(losses.values())
        losses['loss'] = total_loss
        
        return losses
    
    def _compute_losses_with_indices(self, pred_trajs, gt_trajs, indices_list, prefix=''):
        """使用给定的匹配索引计算损失。"""
        losses = {}
        batch_size = pred_trajs.shape[0]
        
        # 初始化各类损失
        pos_loss = 0
        dim_loss = 0
        vel_loss = 0
        yaw_loss = 0
        type_loss = 0
        fp_loss_exist = 0  # 假阳性损失
        loss_acc = 0
        loss_attr = 0
        
        total_objects = 0
        total_fp = 0  # 跟踪假阳性总数
        
        for b in range(batch_size):
            pred_idx, gt_idx = indices_list[b]
            
            # 获取匹配的预测和真实轨迹
            pred_matched = pred_trajs[b, pred_idx]  # [K, TrajParamIndex.END_OF_INDEX]
            gt_matched = gt_trajs[b, gt_idx]  # [K, TrajParamIndex.END_OF_INDEX]
            
            # 计算各项损失
            # 位置损失 (x, y, z) - 使用L1损失
            pos_loss += F.l1_loss(
                pred_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1],
                gt_matched[:, TrajParamIndex.X:TrajParamIndex.Z+1]
            )
            
            # 尺寸损失 (length, width, height) - 使用L1损失
            dim_loss += F.l1_loss(
                pred_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                gt_matched[:, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1]
            )
            
            # 速度损失 (vx, vy) - 使用L1损失
            vel_loss += F.l1_loss(
                pred_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1],
                gt_matched[:, TrajParamIndex.VX:TrajParamIndex.VY+1]
            )
            
            # 加速度损失 (ax, ay) - 使用L1损失
            loss_acc += F.l1_loss(
                pred_matched[:, TrajParamIndex.AX:TrajParamIndex.AY+1],
                gt_matched[:, TrajParamIndex.AX:TrajParamIndex.AY+1]
            )
            
            # 朝向损失 (yaw) - 使用L1损失
            yaw_loss += F.l1_loss(
                pred_matched[:, TrajParamIndex.YAW:TrajParamIndex.YAW+1],
                gt_matched[:, TrajParamIndex.YAW:TrajParamIndex.YAW+1]
            )
            
            # 属性损失 (static, occluded) - 保留BCE损失，因为这是二元属性
            loss_attr += F.binary_cross_entropy_with_logits(
                pred_matched[:, TrajParamIndex.STATIC:TrajParamIndex.OCCLUDED+1],
                gt_matched[:, TrajParamIndex.STATIC:TrajParamIndex.OCCLUDED+1]
            )
            
            # 类型损失 (one-hot encoded object type) - 保留BCE损失，因为这是分类问题
            type_loss += F.binary_cross_entropy_with_logits(
                pred_matched[:, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1],
                gt_matched[:, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1]
            )
            
            # 计算未匹配的预测（假阳性）
            all_pred_idx = set(range(pred_trajs.shape[1]))
            unmatched_pred_idx = list(all_pred_idx - set(pred_idx))
            
            if unmatched_pred_idx:
                # 获取未匹配的预测
                pred_unmatched = pred_trajs[b, unmatched_pred_idx]
                
                # 计算假阳性存在损失 - 所有未匹配的预测应该有 HAS_OBJECT=0
                fp_loss_exist += F.binary_cross_entropy_with_logits(
                    pred_unmatched[:, TrajParamIndex.HAS_OBJECT:TrajParamIndex.HAS_OBJECT+1],
                    torch.zeros_like(pred_unmatched[:, TrajParamIndex.HAS_OBJECT:TrajParamIndex.HAS_OBJECT+1])
                )
                
                # 只对那些预测为有物体的未匹配预测计算额外的假阳性损失
                # 这样可以更强烈地惩罚那些错误地高置信度预测
                fp_high_conf = pred_unmatched[:, TrajParamIndex.HAS_OBJECT] > 0.5
                if fp_high_conf.any():
                    high_conf_fp = pred_unmatched[fp_high_conf]
                    
                    # 对于高置信度的假阳性，我们希望它们的类型预测也是低置信度的
                    # 使用均匀分布作为目标（所有类型概率相等）
                    num_classes = TrajParamIndex.UNKNOWN - TrajParamIndex.CAR + 1
                    uniform_target = torch.ones_like(high_conf_fp[:, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1]) / num_classes
                    
                    # 添加类型损失
                    fp_loss_exist += 0.2 * F.binary_cross_entropy_with_logits(
                        high_conf_fp[:, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1],
                        uniform_target
                    )
                
                total_fp += len(unmatched_pred_idx)
            
            total_objects += len(gt_idx)
        
        # 归一化损失
        if total_objects > 0:
            pos_loss /= total_objects
            dim_loss /= total_objects
            vel_loss /= total_objects
            yaw_loss /= total_objects
            type_loss /= total_objects
            loss_acc /= total_objects
            loss_attr /= total_objects
        
        # 归一化假阳性损失
        if total_fp > 0:
            fp_loss_exist /= total_fp
        else:
            # 如果没有假阳性，我们仍然需要确保梯度流动
            # 创建一个小的常数损失，这样即使没有假阳性，模型也能学习
            fp_loss_exist = torch.tensor(0.01, device=pred_trajs.device)
        
        # 获取层索引并应用层权重
        layer_idx = int(prefix.split('_')[1]) - 4  # 从layer_4开始，所以减4
        layer_weight = self.config.layer_loss_weights[layer_idx] if layer_idx < len(self.config.layer_loss_weights) else 1.0
        
        # 添加到损失字典，应用权重和层权重
        losses[prefix + 'loss_pos'] = pos_loss * self.weight_dict['loss_pos'] * layer_weight
        losses[prefix + 'loss_dim'] = dim_loss * self.weight_dict['loss_dim'] * layer_weight
        losses[prefix + 'loss_vel'] = vel_loss * self.weight_dict['loss_vel'] * layer_weight
        losses[prefix + 'loss_yaw'] = yaw_loss * self.weight_dict['loss_yaw'] * layer_weight
        losses[prefix + 'loss_type'] = type_loss * self.weight_dict['loss_type'] * layer_weight
        losses[prefix + 'loss_acc'] = loss_acc * self.weight_dict['loss_acc'] * layer_weight
        losses[prefix + 'loss_attr'] = loss_attr * self.weight_dict['loss_attr'] * layer_weight
        losses[prefix + 'fp_loss_exist'] = fp_loss_exist * self.weight_dict['fp_loss_exist'] * layer_weight
        
        return losses

    def _compute_matching_cost(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        """计算匹配代价矩阵。
        
        Args:
            pred_trajs: 预测轨迹 [N, TrajParamIndex.END_OF_INDEX]
            gt_trajs: 真实轨迹 [M, TrajParamIndex.END_OF_INDEX]
            
        Returns:
            代价矩阵 [N, M]
        """
        device = pred_trajs.device
        N, M = len(pred_trajs), len(gt_trajs)
        
        if M == 0:
            return torch.zeros((N, 0), device=device)
        
        # 只考虑有效的真实轨迹（HAS_OBJECT=1）
        valid_gt_mask = gt_trajs[:, TrajParamIndex.HAS_OBJECT] > 0.5
        valid_gt_trajs = gt_trajs[valid_gt_mask]
        
        if len(valid_gt_trajs) == 0:
            return torch.zeros((N, 0), device=device)
        
        # 计算代价矩阵
        cost_matrix = torch.zeros((N, len(valid_gt_trajs)), device=device)
        
        for i in range(N):
            for j in range(len(valid_gt_trajs)):
                # 位置代价 - 使用L1损失
                pos_cost = F.l1_loss(
                    pred_trajs[i, TrajParamIndex.X:TrajParamIndex.Z+1],
                    valid_gt_trajs[j, TrajParamIndex.X:TrajParamIndex.Z+1],
                    reduction='sum'
                )
                
                # 尺寸代价 - 使用L1损失
                dim_cost = F.l1_loss(
                    pred_trajs[i, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                    valid_gt_trajs[j, TrajParamIndex.LENGTH:TrajParamIndex.HEIGHT+1],
                    reduction='sum'
                )
                
                # 类型代价（使用交叉熵）
                type_cost = F.binary_cross_entropy_with_logits(
                    pred_trajs[i, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1],
                    valid_gt_trajs[j, TrajParamIndex.CAR:TrajParamIndex.UNKNOWN+1],
                    reduction='sum'
                )
                
                # 综合代价
                cost = (
                    pos_cost * self.weight_dict['loss_pos'] +
                    dim_cost * self.weight_dict['loss_dim'] +
                    type_cost * self.weight_dict['loss_type']
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix

def calculate_trajectory_bev_iou(traj1: torch.Tensor, traj2: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Calculate bird's eye view IoU between trajectories.
    
    Args:
        traj1: First trajectory [B, TrajParamIndex.END_OF_INDEX]
        traj2: Second trajectory [B, TrajParamIndex.END_OF_INDEX]
        frames: Frame timestamps [T]
        
    Returns:
        IoU value
    """
    # TODO: Implement BEV IoU calculation
    return torch.zeros(1, device=traj1.device)

def calculate_trajectory_bev_iou2(traj1: torch.Tensor, traj2: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Calculate bird's eye view IoU between trajectories using improved method.
    
    Args:
        traj1: First trajectory [B, TrajParamIndex.END_OF_INDEX]
        traj2: Second trajectory [B, TrajParamIndex.END_OF_INDEX]
        frames: Frame timestamps [T]
        
    Returns:
        IoU value
    """
    # TODO: Implement improved BEV IoU calculation
    return torch.zeros(1, device=traj1.device)

def calculate_trajectory_distance_score(traj1: torch.Tensor, traj2: torch.Tensor) -> torch.Tensor:
    """Calculate distance between trajectories.
    
    Args:
        traj1: First trajectory [TrajParamIndex.END_OF_INDEX]
        traj2: Second trajectory [TrajParamIndex.END_OF_INDEX]
        
    Returns:
        Distance value
    """
    # calculate distance between two points
    pos1 = traj1[[TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]]
    pos2 = traj2[[TrajParamIndex.X, TrajParamIndex.Y, TrajParamIndex.Z]]
    dist = torch.norm(pos1 - pos2)
    
    # Use average size of both objects to normalize distance
    size1 = traj1[[TrajParamIndex.LENGTH, TrajParamIndex.WIDTH, TrajParamIndex.HEIGHT]]
    size2 = traj2[[TrajParamIndex.LENGTH, TrajParamIndex.WIDTH, TrajParamIndex.HEIGHT]]
    size_diff = torch.norm(size1 - size2)
    
    
    # Normalized distance score that decays with distance relative to object size
    dist_score = torch.exp(-(dist + size_diff) / (size_diff + 1e-6))
    return dist_score
