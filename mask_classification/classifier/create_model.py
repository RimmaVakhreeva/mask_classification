import torch.nn
import torch.optim

import pretrainedmodels

__all__ = ['create_model']


class ClassifierWrapper(torch.nn.Module):
    def __init__(self,
                 backbone_module: torch.nn.Module,
                 input_channels: int,
                 num_classes: int,
                 freeze_backbone: bool = False):
        super().__init__()
        self.backbone_module = backbone_module
        if freeze_backbone:
            self.backbone_module.requires_grad = False
            self.backbone_module.eval()
        self.classifier_layers = [
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(input_channels, num_classes)]
        self.classifier_layers = torch.nn.Sequential(*self.classifier_layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.backbone_module(x)
        x = self.classifier_layers(x)
        if not self.training:
            x = x.sigmoid()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def create_model(pretrained, **kwargs) -> torch.nn.Module:
    backbone_type = kwargs.pop('backbone_type')
    if backbone_type in pretrainedmodels.__dict__:
        backbone = pretrainedmodels.__dict__[backbone_type](pretrained=pretrained)
        backbone.forward = backbone.features
    else:
        assert False, f"{backbone_type} not found in pretrainedmodels"
    return ClassifierWrapper(backbone_module=backbone,
                             input_channels=backbone.last_linear.in_features, **kwargs)
