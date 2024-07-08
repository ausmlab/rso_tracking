base_lr = 0.004
img_size = 640 #640
random_size_range = (320, 640) #(320, 640)
work_dir = './work_dirs/FAI_yolox_nano_2seqs'
path_to_rso_tracking = '/nas2/YJ/git/rso_tracking/'
DATA_ROOT = path_to_rso_tracking + 'data/DET_COCO_STYLE_TWOs/ADDCURR/'  # parent folder of 'train' and 'test'
checkpoint = path_to_rso_tracking + 'pretrained_weights/yolox_nano_mmdet.pth' 

#################################################################################################################
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    img_size,
                    img_size,
                ),
                type='RandomResize'),
            dict(crop_size=(
                img_size,
                img_size,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    img_size,
                    img_size,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
]
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook', score_thr=0.3) ##<--------------------------
    )
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

interval = 10
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        act_cfg=dict(type='Swish'),
        deepen_factor=0.33,
        init_cfg=dict(
            checkpoint=checkpoint,
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type='CSPDarknet',
        use_depthwise=True,
        widen_factor=0.25),
    bbox_head=dict(
        act_cfg=dict(type='Swish'),
        feat_channels=64,
        in_channels=64,
        loss_bbox=dict(
            eps=1e-16,
            loss_weight=5.0,
            mode='square',
            reduction='sum',
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_classes=1,
        stacked_convs=2,
        strides=(
            8,
            16,
            32,
        ),
        type='YOLOXHead',
        use_depthwise=True),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=10,
                random_size_range=random_size_range,
                size_divisor=32,
                type='BatchSyncRandomResize'),
        ],
        pad_size_divisor=32,
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(type='Swish'),
        in_channels=[
            64,
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=1,
        out_channels=64,
        type='YOLOXPAFPN',
        upsample_cfg=dict(mode='nearest', scale_factor=2),
        use_depthwise=True),
    test_cfg=dict(nms=dict(iou_threshold=0.2, type='nms'), score_thr=0.001),  ###<-----------------------
    train_cfg=dict(assigner=dict(center_radius=2.5, type='SimOTAAssigner')),
    type='YOLOX')

optim_wrapper = dict(
    optimizer=dict(lr=base_lr, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=150,
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
resume = False

test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file= DATA_ROOT + 'test/annotation_coco.json',
        backend_args=None,
        data_prefix=dict(img='./'),
        data_root=DATA_ROOT + 'test',
        metainfo=dict(classes=('RSO', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                img_size, 
                img_size, 
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    img_size, 
                    img_size, 
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=DATA_ROOT + 'test/annotation_coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    iou_thrs=[0.01], ###<--------------- #, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95],
    proposal_nums=(
        100, # 100
        1, # 1, 300
        10, # 10, 1000
    ),
    type='CocoMetric')

train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=10)

train_dataloader = dict(
    batch_sampler=None,
    batch_size=32,
    dataset=dict(
        ann_file=DATA_ROOT + 'train/annotation_coco.json',
        backend_args=None,
        data_prefix=dict(img='./'),
        data_root=DATA_ROOT + 'train',
        filter_cfg=dict(filter_empty_gt=True, min_size=0),  ###<--------------------------------
        metainfo=dict(classes=('RSO', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),

        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    img_size,
                    img_size,
                ),
                max_cached_images=20,
                pad_val=114.0,
                random_pop=False,
                type='CachedMosaic'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    img_size*2,
                    img_size*2,
                ),
                type='RandomResize'),
            dict(crop_size=(
                img_size,
                img_size,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    img_size,
                    img_size,
                ),
                type='Pad'),
            dict(
                img_scale=(
                    img_size,
                    img_size,
                ),
                max_cached_images=10,
                pad_val=(
                    114,
                    114,
                    114,
                ),
                prob=0.5,
                random_pop=False,
                ratio_range=(
                    1.0,
                    1.0,
                ),
                type='CachedMixUp'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file=DATA_ROOT + 'test/annotation_coco.json',
        backend_args=None,
        data_prefix=dict(img='./'),
        data_root=DATA_ROOT + 'test',
        metainfo=dict(classes=('RSO', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                img_size,
                img_size,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    img_size,
                    img_size,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = dict(
    ann_file=DATA_ROOT + 'test/annotation_coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    iou_thrs=[0.01],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
