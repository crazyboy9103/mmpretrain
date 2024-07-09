_base_ = [
    '../_base_/models/swin_transformer/tiny_448.py',
    '../_base_/datasets/hanrim_bs8.py',
    '../_base_/schedules/hanrim_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(img_size=448), 
    head=dict(num_classes={{_base_.data_preprocessor.num_classes}})
)
# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'swin-tiny', 'hanrim']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)