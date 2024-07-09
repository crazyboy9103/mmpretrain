# Refers to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification

_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_large_imagenet.py',
    '../_base_/datasets/imagenet_bs128_mbv3.py',
    '../_base_/default_runtime.py',
]

model = dict(
    pretrained=None
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type='RMSprop',
        lr=0.064,
        alpha=0.9,
        momentum=0.9,
        eps=0.0316,
        weight_decay=1e-5))

param_scheduler = dict(type='StepLR', by_epoch=True, step_size=2, gamma=0.973)

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['cla', 'mobilenetv3large', 'imagenet']),)
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)
