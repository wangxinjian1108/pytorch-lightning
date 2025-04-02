from typing import List, Dict
from xinnovation.src.core import (SourceCameraId, CameraType, CameraParamIndex, EgoStateIndex, TrajParamIndex)
from xinnovation.examples.detector4D.sparse4d_dataset import CameraGroupConfig

# ============================== 1. Base Config ==============================


# ============================== 2. Model Config ==============================
# ├── LIGHTNING_MODULE
# │   ├── OPTIMIZERS         # 优化器 (SGD, AdamW, Lion)
# │   ├── SCHEDULERS         # 学习率策略 (StepLR, CosineAnnealingLR)
# │   ├── LOSSES             # 损失函数 (FocalLoss, SmoothL1, DiceLoss)
# │   ├── DETECTORS          # 检测器 (YOLO, SSD, FasterRCNN)
# │   │   ├── IMAGE_FEATURE_EXTRACTOR # 图像特征提取器 (ResNet, SwinTransformer, PointNet++)
# │   │   │   ├── BACKBONES          # 骨干网络 (ResNet, SwinTransformer, PointNet++)
# │   │   │   ├── NECKS              # 特征融合 (FPN, BiFPN, ASFF)
# │   │   ├── HEADS              # 任务头 (DetectionHead, ClassificationHead)
# │   │   ├── PLUGINS              # 插件 (AnchorGenerator, Anchor3DGenerator, SineEncoding, LearnedEncoding)
# │   │   │   ├── ANCHOR_GENERATOR  # 锚框生成器 (AnchorGenerator, Anchor3DGenerator)
# │   │   │   ├── POS_ENCODING       # 位置编码 (SineEncoding, LearnedEncoding)
# │   │   │   ├── ATTENTION          # 注意力机制 (SelfAttention, CBAM, TransformerBlock)
# │   │   │   ├── NORM_LAYERS        # 归一化层 (BatchNorm, LayerNorm, GroupNorm)
# │   │   │   ├── FEEDFORWARD_NETWORK # 前馈网络 (AsymmetricFFN)
# │   │   │   ├── DROPOUT            # 丢弃层 (Dropout)

query_dim = 256
dropout = 0.1
num_groups = 8
num_decoder = 6
num_classes = TrajParamIndex.END_OF_INDEX - TrajParamIndex.HAS_OBJECT
use_temp_attention = False
with_quality_estimation = False
def repvgg_backbone(name: str="a1", scales_to_drop: List[int]=[2, 4], use_pretrained: bool=True):
    return dict(
        type="ImageFeatureExtractor",
        backbone=f'repvgg_{name}',
        scales_to_drop=scales_to_drop,
        use_pretrained=use_pretrained
    )

def fpn_neck(in_channels: List[int]=[256, 512, 1024]):
    return dict(
        type="FPN",
        in_channels=in_channels,
        out_channels=query_dim,
        extra_blocks=0,
        relu_before_extra_convs=False
    )

lightning_module = dict(
    type="Sparse4DModule",
    scheduler=dict(
        type="StepLRScheduler",
        step_size=10,
        gamma=0.1
    ),
    optimizer=dict(
        type="AdamWOptimizer",
        lr=0.001,
        weight_decay=0.0001
    ),
    loss=dict(
        type="Sparse4DLossWithDAC",
        layer_loss_weights=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    detector=dict(
        type="Sparse4DDetector",
        camera_groups=dict(
            front_stereo_camera=[SourceCameraId.FRONT_LEFT_CAMERA, SourceCameraId.FRONT_RIGHT_CAMERA],
            short_focal_length_camera=[SourceCameraId.FRONT_CENTER_CAMERA],
            rear_camera=[SourceCameraId.REAR_LEFT_CAMERA, SourceCameraId.REAR_RIGHT_CAMERA]
        ),
        anchor_encoder=dict(
            type="AnchorEncoder",
            pos_embed_dim=128,
            dim_embed_dim=32,
            orientation_embed_dim=32,
            vel_embed_dim=64,
            embed_dim=query_dim,
            anchor_generator_config=dict(
                type="Anchor3DGenerator",
                front_type="div_x",
                back_type="div_x",
                front_params=dict(alpha=0.4, beta=10.0, order=2.0),
                back_params=dict(alpha=0.35, beta=10.0, order=2.0),
                front_min_spacing=2.0,
                front_max_distance=200.0,
                back_min_spacing=2.0,
                back_max_distance=100.0,
                left_y_max=3.75 * 2,
                right_y_max=3.75 * 2,
                y_interval=3.75,
                z_value=0.2,
                anchor_size=(5.0, 2.0, 1.5) # (length, width, height)
            )
        ),
        feature_extractors=dict(
            # Ensure all the feature extractors have the same FPN levels
            front_stereo_camera=dict(
                type="FPNImageFeatureExtractor",
                backbone=repvgg_backbone(name="a2", scales_to_drop=[2, 4], use_pretrained=True),
                neck=fpn_neck()
            ),
            short_focal_length_camera=dict(
                type="FPNImageFeatureExtractor",
                backbone=repvgg_backbone(name="a1", scales_to_drop=[2, 4], use_pretrained=True),
                neck=fpn_neck()
            ),
            rear_camera=dict(
                type="FPNImageFeatureExtractor",
                backbone=repvgg_backbone(name="a1", scales_to_drop=[2, 4], use_pretrained=True),
                neck=fpn_neck()
            )
        ),
        decoder_op_orders=[[
                "temp_attention",
                "self_attention",
                "mts_feature_aggregator",
                "ffn",
                "refine",
            ]
            for _ in range(num_decoder)
        ],
        mts_feature_aggregator=dict(
            type="MultiviewTemporalSpatialFeatureAggregator",
            query_dim=query_dim,
            num_learnable_points=8,
            learnable_points_range=3.0,
            sequence_length=10,
            temporal_weight_decay=0.5,
            camera_nb=7,
            fpn_levels=3,
            residual_mode="cat"
        ),
        self_attention=dict(
            type="DecoupledMultiHeadAttention",
            query_dim=query_dim,
            num_heads=num_groups,
            dropout=dropout,
            post_norm=dict(type="LayerNorm", eps=1e-6, normalized_shape=query_dim),
        ),
        ffn=dict(
            type="AsymmetricFFN",
            pre_norm=dict(type="LayerNorm", eps=1e-6, normalized_shape=query_dim * 2),
            post_norm=dict(type="LayerNorm", eps=1e-6, normalized_shape=query_dim),
            in_channels=query_dim * 2,
            embed_dims=query_dim,
            feedforward_channels=query_dim * 4,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
            ffn_drop=0.1,
            add_identity=True
        ),
        refine=dict(
            type="TrajectoryRefiner",
            query_dim=query_dim,
            hidden_dim=query_dim,
            with_quality_estimation=with_quality_estimation,
        ),
        temp_attention=dict(
            type="DecoupledMultiHeadAttention",
            query_dim=query_dim,
            num_heads=num_groups,
            dropout=dropout,
            post_norm=None
        ) if use_temp_attention else None
    )
)
# ============================== 3. Data Config ==============================
lightning_data_module = dict(
    type="Sparse4DDataModule",
    train_list="/home/xinjian/Code/pytorch-lightning/train_clips.txt",
    val_list="/home/xinjian/Code/pytorch-lightning/val_clips.txt",
    batch_size=1,
    num_workers=4,
    sequence_length=10,
    shuffle=True,
    persistent_workers=True,
    pin_memory=True,
    camera_groups=[CameraGroupConfig.front_stereo_camera_group(), CameraGroupConfig.short_focal_length_camera_group(), CameraGroupConfig.rear_camera_group()]
)


# ============================== 4. Trainer Config ==============================


# ============================== 5. Evaluation Config ==============================



if __name__ == "__main__":
    from xinnovation.src.core.config import Config
    from easydict import EasyDict as edict
    import os
    
    # Get the absolute path to this file
    current_file = os.path.abspath(__file__)
    cfg = Config.from_file(current_file)
    config = Config.from_file(current_file).to_dict()
    cfg.save_to_file(current_file.replace(".py", ".json"))
    cfg.save_to_file(current_file.replace(".py", ".yaml"))
