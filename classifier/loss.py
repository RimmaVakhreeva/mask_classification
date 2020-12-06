import torch
import torch.nn.functional as F

__all__ = ['BinaryCrossEntropy']


class BinaryCrossEntropy(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(BinaryCrossEntropy, self).__init__()
        self._loss = torch.nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, inputs, targets):
        return self._loss(inputs, targets.type_as(inputs))
