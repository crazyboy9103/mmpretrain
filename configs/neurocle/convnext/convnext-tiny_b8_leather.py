_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/leather_bs8.py',
    '../_base_/schedules/leather_swin.py',
    '../_base_/default_runtime.py',
]

# EMAHook does not seem to work well for fine-tuning
# runtime setting
# custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

model = dict(
    head=dict(
        num_classes={{_base_.data_preprocessor.num_classes}},
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'convnext-tiny', 'leather']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)