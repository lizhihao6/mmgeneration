dataset_type = 'ganISPDataset'
domain_a = 'raw'  # set by user
domain_b = 'rgb'  # set by user
dataroot_a = '/lzh/datasets/multiRAW/iphone_xsmax/raw'
train_dataroot_b = '/shared/cityscapes/leftImg8bit/train'
test_dataroot_b = '/shared/cityscapes/leftImg8bit/test'
train_split_a = '/lzh/datasets/multiRAW/iphone_xsmax/train.txt'
test_split_a = '/lzh/datasets/multiRAW/iphone_xsmax/train.txt'
split_b = None

train_pipeline = [
    dict(
        type='LoadRAWFromFile',
        key=f'img_{domain_a}'),
    dict(
        type='Demosaic',
        key=f'img_{domain_a}'),
    dict(
        type='RAWNormalize',
        blc=528,
        saturate=4095,
        key=f'img_{domain_a}'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}'],
        scale=(2048, 1539),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        crop_size=(512, 512),
        random_crop=True),
    dict(type='Flip', keys=[f'img_{domain_a}'], direction='horizontal'),
    dict(type='Flip', keys=[f'img_{domain_b}'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_b}'],
        to_rgb=True,
        mean=[0, 0, 0],
        std=[1, 1, 1]),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]

test_pipeline = [
 dict(
        type='LoadRAWFromFile',
        key=f'img_{domain_a}'),
    dict(
        type='Demosaic',
        key=f'img_{domain_a}'),
    dict(
        type='RAWNormalize',
        blc=528,
        saturate=4095,
        key=f'img_{domain_a}'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(2048, 1024),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=[f'img_{domain_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{domain_b}'],
        to_rgb=True,
        mean=[0, 0, 0],
        std=[1, 1, 1]),
    dict(type='ImageToTensor', keys=[f'img_{domain_a}', f'img_{domain_b}']),
    dict(
        type='Collect',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    drop_last=True,
    val_samples_per_gpu=1,
    val_workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        dataroot_a=dataroot_a,
        dataroot_b=train_dataroot_b,
        split_a=train_split_a,
        split_b=split_b,
        pipeline=train_pipeline,
        domain_a=domain_a,
        domain_b=domain_b),
    val=dict(
        type=dataset_type,
        dataroot_a=dataroot_a,
        dataroot_b=test_dataroot_b,
        split_a=test_split_a,
        split_b=split_b,
        domain_a=domain_a,
        domain_b=domain_b,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dataroot_a=dataroot_a,
        dataroot_b=test_dataroot_b,
        split_a=test_split_a,
        split_b=split_b,
        domain_a=domain_a,
        domain_b=domain_b,
        pipeline=test_pipeline))
