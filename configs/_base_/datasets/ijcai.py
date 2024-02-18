# dataset settings
dataset_type = 'DOTADataset'
data_root = '/root/data/IJCAI/split_ss_ijcai/'

CLASSES = ('pressure', 'umbrella', 'lighter', 'OCbottle', 'glassbottle',
            'battery', 'metalbottle', 'knife', 'electronicequipment')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes = CLASSES,
        ann_file=data_root + 'train_track2/annfiles/',
        img_prefix=data_root + 'train_track2/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = CLASSES,
        ann_file=data_root + 'train_track2/annfiles/',
        img_prefix=data_root + 'train_track2/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = CLASSES,
        ann_file=data_root + 'test1_phase1/images/',
        img_prefix=data_root + 'test1_phase1/images/',
        # ann_file=data_root + 'trainval/annfiles/',
        # img_prefix=data_root + 'trainval/images/',
        pipeline=test_pipeline))
