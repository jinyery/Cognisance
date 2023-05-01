######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(m_type='resnet101'):
    # create various resnet models
    if m_type == 'resnet18':
        model = models.resnet18(weights=None)
    elif m_type == 'resnet50':
        model = models.resnet50(weights=None)
    elif m_type == 'resnet101':
        model = models.resnet101(weights=None)
    elif m_type == 'resnext50':
        model = models.resnext50_32x4d(weights=None)
    elif m_type == 'resnext101':
        model = models.resnext101_32x8d(weights=None)
    else:
        raise ValueError('Wrong Model Type')
    model.fc = nn.ReLU()
    return model