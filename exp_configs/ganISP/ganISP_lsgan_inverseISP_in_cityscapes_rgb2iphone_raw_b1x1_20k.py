_base_ = [
    '../_base_/models/ganISP/ganISP_lsgan_inverseISP.py',
    '../_base_/datasets/cityscapes_rgb2iphone_raw_512x512.py',
    '../_base_/default_runtime.py'
]
train_cfg = dict(buffer_size=50)
test_cfg = None
domain_a = 'raw'
domain_b = 'rgb'
model = dict(
    default_domain=domain_b,
    reachable_domains=[domain_a, domain_b],
    related_domains=[domain_a, domain_b],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_b}',
                target=f'real_{domain_b}',
            ),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.1,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_a}',
                target=f'real_{domain_a}',
            ),
            reduction='mean')
    ])

optimizer = dict(
    generator=dict(type='AdamW', lr=5e-4),
    discriminators=dict(type='Adam', lr=5e-5, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=10000, interval=400)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=[f'real_{domain_a}', f'fake_{domain_b}', f'cycle_{domain_a}', f'real_{domain_b}', f'fake_{domain_a}', f'cycle_{domain_b}'],
        rerange=False,
        bgr2rgb=False,
        interval=200)
]

runner = None
find_unused_parameters = True
use_ddp_wrapper = True
total_iters = 20000
workflow = [('train', 1)]
exp_name = 'ganISP_cityscapes_rgb2iphone_raw_without_identity'
work_dir = f'./work_dirs/experiments/{exp_name}'
num_images = 20

evaluation = dict(
    type='TranslationEvalHook',
    target_domain=domain_a,
    interval=10000,
    metrics=[
        dict(type='FID', num_images=num_images, bgr2rgb=False),
        dict(
            type='IS',
            num_images=num_images,
            image_shape=(3, 256, 256),
            inception_args=dict(type='pytorch'))
    ],
    best_metric=['fid', 'is'])
