_base_ = [
    '../_base_/models/swin_transformer/tiny_448.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None
)

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'swin-tiny', 'imagenet']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)