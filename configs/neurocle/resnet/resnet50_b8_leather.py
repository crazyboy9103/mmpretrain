_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/leather_bs8.py',
    '../_base_/schedules/leather.py', '../_base_/default_runtime.py'
]

model = dict(
    head=dict(num_classes={{_base_.data_preprocessor.num_classes}})
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'resnet50', 'leather']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)
