_base_ = [
    '../_base_/models/regnet/regnetx_4.0gf.py',
    '../_base_/datasets/leather_bs8.py',
    '../_base_/schedules/leather.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)

# schedule settings

# sgd with nesterov, base ls is 0.8 for batch_size 1024,
optim_wrapper = dict(optimizer=dict(lr=0.4))

# runtime settings

# Precise BN hook will update the bn stats, so this hook should be executed
# before CheckpointHook(priority of 'VERY_LOW') and
# EMAHook(priority of 'NORMAL') So set the priority of PreciseBNHook to
# 'ABOVENORMAL' here.
# custom_hooks = [
#     dict(
#         type='PreciseBNHook',
#         num_samples=8192,
#         interval=1,
#         priority='ABOVE_NORMAL')
# ]

# model settings
model = dict(
    backbone=dict(type='RegNet', arch='regnetx_4.0gf'),
    head=dict(in_channels=1360, num_classes={{_base_.data_preprocessor.num_classes}}))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'regnetx-4.0gf', 'leather']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)