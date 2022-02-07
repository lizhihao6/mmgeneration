_domain_a = None  # set by user
_domain_b = None  # set by user
model = dict(
    type='ganISP',
    generator=dict(
        type='inverseISP'),
    discriminator=dict(
        type='HistAwareDiscriminator'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    default_domain=None,  # set by user
    reachable_domains=None,  # set by user
    related_domains=None,  # set by user
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{_domain_a}', target=f'real_{_domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{_domain_b}',
                target=f'real_{_domain_b}',
            ),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{_domain_a}', target=f'real_{_domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{_domain_b}', target=f'real_{_domain_b}'),
            reduction='mean')
    ])
train_cfg = dict(buffer_size=50)
test_cfg = None
