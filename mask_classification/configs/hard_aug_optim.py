from functools import partial
from pathlib import Path

import cv2
import torch
import numpy as np
import albumentations as A
import albumentations.pytorch as AP

from mask_classification.classifier.dataset import FaceMaskDataset
from mask_classification.classifier.create_model import create_model
from mask_classification.classifier.loss import BinaryCrossEntropy
from mask_classification.classifier.utils import load_images, seed_everything_deterministic

__all__ = ['base_kwargs', 'trainer_kwargs', 'lightning_module_kwargs']

seed_everything_deterministic(42)

base_kwargs = dict(
    log_dir=Path('./logs/baseline'))
trainer_kwargs = dict(
    gpus=[1],
    max_epochs=20,
    val_check_interval=1.,
    precision=16,
    amp_backend='native',
    distributed_backend='ddp'
)
lightning_module_kwargs = dict(
    batch_size=128,
    num_workers=8,
    optimizer_cls=partial(torch.optim.SGD, lr=1e-3),
    scheduler_cls=dict(scheduler=partial(torch.optim.lr_scheduler.StepLR,
                                         step_size=trainer_kwargs['max_epochs'] // 3, gamma=0.1),
                       interval='epoch',
                       frequency=1)
)

width, height = 112, 112

model_type = 'classifier'
backbone_type = 'resnet18'
pretrained = 'imagenet'
freeze_backbone = False
train_test_split = 0.7

wandb_kwargs = dict(
    project='classifier',
    name=None,
    tags=['baseline'],
    group='baseline'
)

images_with_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/main_images_with_mask')
images_without_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/without_mask')

norm_images_with_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/norm_with_mask')
norm_images_without_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/norm_without_mask')

train_images_data = [*load_images(images_with_mask, masks_folder_name='general_with_mask'),
                     *load_images(images_without_mask, masks_folder_name='general_with_mask')]
test_images_data = [*load_images(norm_images_with_mask, masks_folder_name='norm_with_mask'),
                    *load_images(norm_images_without_mask, masks_folder_name='norm_with_mask')]

loss = 'loss/bce', BinaryCrossEntropy()

scale_min = 0.6
scale_max = 0.99

train_transfoms = A.Compose([
    A.LongestMaxSize(max_size=height),
    A.PadIfNeeded(min_width=width, min_height=height, value=(128, 128, 128), border_mode=0),

    A.HorizontalFlip(),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.OneOf(transforms=[
        A.Downscale(scale_min=scale_min, scale_max=scale_max, interpolation=cv2.INTER_NEAREST, p=1),
        A.Downscale(scale_min=scale_min, scale_max=scale_max, interpolation=cv2.INTER_LINEAR, p=1),
        A.Downscale(scale_min=scale_min, scale_max=scale_max, interpolation=cv2.INTER_LANCZOS4, p=1),
    ], p=1),
    A.RandomBrightnessContrast(brightness_limit=(0., 0.5), contrast_limit=(0., 0.4), p=0.5),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=1.),
        A.MedianBlur(blur_limit=3, p=1.),
        A.Blur(blur_limit=3, p=1.),
    ], p=0.2),
    A.HueSaturationValue(p=0.3),

    A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.),
    AP.transforms.ToTensorV2()
])

test_transfoms = A.Compose([
    A.LongestMaxSize(max_size=height),
    A.PadIfNeeded(min_width=width, min_height=height, value=(128, 128, 128), border_mode=0),
    A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.),
    AP.transforms.ToTensorV2()
])

# Export part
train_dataset = FaceMaskDataset(train_images_data, transform=train_transfoms)
val_dataset = FaceMaskDataset(test_images_data, transform=test_transfoms)

model = create_model(backbone_type=backbone_type,
                     pretrained=pretrained,
                     num_classes=1,
                     freeze_backbone=freeze_backbone)

lightning_module_kwargs.update(dict(model=model, loss=loss,
                                    train_ds=train_dataset, val_ds=val_dataset))
