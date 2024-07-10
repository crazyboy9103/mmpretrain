# In small and tiny arch, remove drop path and EMA hook comparing with the
# original config
_base_ = [
    '../_base_/datasets/hanrim_bs8.py',
    '../_base_/schedules/hanrim_swin.py',
    '../_base_/default_runtime.py'
]

image_size=_base_.train_pipeline[1]['scale'][0]
# model settings
model = dict(
    type='ImageClassifier',
    pretrained='https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth', 
    backbone=dict(
        type='VisionTransformer',
        arch='deit-small',
        img_size=image_size,
        patch_size=16),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes={{_base_.data_preprocessor.num_classes}},
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# data settings
train_dataloader = dict(batch_size=8)

# schedule settings
optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'deit-small', 'hanrim']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)