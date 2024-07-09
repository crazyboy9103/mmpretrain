_base_ = [
    '../_base_/models/efficientnet_v2/efficientnetv2_b0.py',
    '../_base_/datasets/leather_bs8.py',
    '../_base_/schedules/leather.py',
    '../_base_/default_runtime.py',
]

# model setting
model = dict(backbone=dict(arch='b3'), head=dict(in_channels=1536, num_classes={{_base_.data_preprocessor.num_classes}}))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'efnetv2b3', 'leather']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)
