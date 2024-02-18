_base_ = [
    '/root/mmrotate/configs/_base_/datasets/dotav1.py', 
    '/root/mmrotate/configs/_base_/schedules/schedule_1x.py',
    '/root/mmrotate/configs/_base_/default_runtime.py'
]

angle_version = 'le90'
model = dict(
    type='KDRotatedSingleStageDetector',
    teacher_config='/root/mmrotate/configs/nsd/DOTA/rotated_retinanet/rotated_retinanet_obb_r101_fpn_1x_dota_le90.py',
    teacher_ckpt='/root/mmrotate/train_nsd/DOTA/rotated_retinanet/rotated_retinanet_obb_r101_fpn_1x_dota_le90/latest.pth',
    # teacher_ckpt='/root/mmrotate/train_nsd/DOTA/baseline/rotated_retinanet/rotated_retinanet_obb_r101_fpn_1x_dota_le90/latest.pth',
    output_feature=True,
    fpnkd_cfg=dict(
        feature_kd=True,
        type='neck-adapt,neck-decouple,mask-neck-gt,pixel-wise',
        neck_in_channels=[256,256,256,256,256],
        neck_out_channels=[256,256,256,256,256],
        bb_in_channels=[512,1024,2048],
        bb_out_channels=[512,1024,2048],
        bb_indices=(1,2,3),
        hint_neck_w=0,
        hint_neck_back_w=12, 
        hint_bb_w=0,
        hint_bb_back_w=0,
        head_cls_w=2,
        head_cls_back_w=2,
        head_cls_T=1,
        head_cls_back_T=1),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='AD_LD_CD_RRetinaHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        angle_coder=dict(
            type='AngleCLSCoder',
            angle_range=90,
            omega=3),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_angle=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        use_angle_negsample=True,
        loss_ad=dict(
            type='NAPSingleLoss',
            alpha=0.0, 
            beta=8.0, 
            T=4.0, 
            warmup=3,
            loss_weight=1.0,
            use_sigmoid=True,
            use_sample='negn'),
        loss_cd=dict(
            type='NAPSingleLoss',
            alpha=0.0, 
            beta=8.0, 
            T=4.0, 
            warmup=3,
            loss_weight=1.8,
            use_sigmoid=True,
            use_sample='negn'),
        ld_sample="neg",
        loss_ld=dict(type='L1Loss', loss_weight=0.9)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
# fp16 = dict(loss_scale='dynamic')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    # samples_per_gpu=1,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
custom_hooks = [dict(type='SetEpochInfoHook')]
# evaluation = dict(interval=12, metric='mAP')
