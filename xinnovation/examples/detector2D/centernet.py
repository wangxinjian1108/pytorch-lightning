import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from xinnovation.src.core import HEADS, LOSSES
from xinnovation.src.components.lightning_module.losses.classification import FocalLoss
from xinnovation.src.components.lightning_module.losses.regression import SmoothL1Loss

__all__ = ["MultiLevelCenterNetHead", "CenterNetLoss"]

@HEADS.register_module()
class MultiLevelCenterNetHead(nn.Module):
    def __init__(self, num_classes: int, 
                 fpn_levels: int, 
                 fpn_in_channel: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.hm_heads = nn.ModuleList([self._make_head(fpn_in_channel, hidden_dim, num_classes) for _ in range(fpn_levels)])  # P3,P4,P5
        self.wh_heads = nn.ModuleList([self._make_head(fpn_in_channel, hidden_dim, 2) for _ in range(fpn_levels)])
        self.offset_heads = nn.ModuleList([self._make_head(fpn_in_channel, hidden_dim, 2) for _ in range(fpn_levels)])

    def _make_head(self, fpn_in_channel, hidden_dim, out_channels):
        return nn.Sequential(
            nn.Conv2d(fpn_in_channel, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1)
        )

    def forward(self, features):  # features = [P3, P4, P5]
        outputs = []
        for i, f in enumerate(features):
            hm = self.hm_heads[i](f)
            wh = self.wh_heads[i](f)
            offset = self.offset_heads[i](f)
            outputs.append((hm, wh, offset))
        return outputs
    

@LOSSES.register_module()
class CenterNetLoss(nn.Module):
    def __init__(self, num_classes: int, fpn_levels: int, fpn_in_channel: int, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_levels = fpn_levels
        self.fpn_in_channel = fpn_in_channel
        self.hidden_dim = hidden_dim

    def forward(self, preds:List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                targets_per_level:List[Dict], 
                strides:List[int]):
        """
        preds: list of (heatmap, wh, offset) from P3/P4/P5, each shape:
            heatmap: (B, C, H, W)
            wh: (B, 2, H, W)
            offset: (B, 2, H, W)
        
        targets_per_level: list of target dicts per level, each dict:
            {
                'heatmap': (B, C, H, W),
                'wh': (B, max_objs, 2),
                'offset': (B, max_objs, 2),
                'indices': (B, max_objs),
                'mask': (B, max_objs),
            }
        """
        total_hm_loss, total_wh_loss, total_off_loss = 0, 0, 0
        for i, (pred, target, stride) in enumerate(zip(preds, targets_per_level, strides)):
            pred_hm, pred_wh, pred_off = pred
            gt_hm = target['heatmap']  # (B, C, H, W)

            pred_hm = torch.sigmoid(pred_hm)
            hm_loss = self._focal_loss(pred_hm, gt_hm)

            # wh + offset loss
            B, max_objs = target['wh'].shape[:2]
            indices = target['indices']  # (B, max_objs)
            mask = target['mask'].unsqueeze(2)  # (B, max_objs, 1)

            pred_wh = _gather_feat(pred_wh.permute(0, 2, 3, 1).contiguous(), indices)  # (B, max_objs, 2)
            pred_off = _gather_feat(pred_off.permute(0, 2, 3, 1).contiguous(), indices)  # (B, max_objs, 2)

            wh_loss = F.l1_loss(pred_wh * mask, target['wh'] * mask, reduction='sum') / (mask.sum() + 1e-6)
            off_loss = F.l1_loss(pred_off * mask, target['offset'] * mask, reduction='sum') / (mask.sum() + 1e-6)

            total_hm_loss += hm_loss
            total_wh_loss += wh_loss
            total_off_loss += off_loss

        return {
            "heatmap_loss": total_hm_loss,
            "wh_loss": total_wh_loss,
            "offset_loss": total_off_loss,
            "total_loss": total_hm_loss + total_wh_loss + total_off_loss,
        }