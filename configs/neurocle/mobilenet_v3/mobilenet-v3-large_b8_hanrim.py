# Refers to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification

_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_large_imagenet.py',
    '../_base_/datasets/hanrim_bs8.py',
    '../_base_/schedules/hanrim.py',
    '../_base_/default_runtime.py',
]

model = dict(
    head=dict(
        num_classes={{_base_.data_preprocessor.num_classes}}
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'mobilenetv3large', 'hanrim']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)