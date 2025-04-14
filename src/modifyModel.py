import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
import math
import numpy as np
from kernels import *
import torchvision.models as models

def modifyModel(model):
    # Iterate over all named layers in the model
    for name, module in model.named_children():
        # If the module is a Linear layer, replace it with your custom FC layer
        #print(module)
        #print(name)
        if isinstance(module, nn.Linear):
            setattr(model, name, OpticalFC(module.weight, module.bias))
        elif isinstance(module, nn.Conv2d):
            if(module.groups != 1):
                raise Exception('Groups neq 1 not supported')
            if(module.padding_mode != "zeros"):
                raise Exception('padding_mode neq \'zeros\' not supported')
            setattr(model, name, OpticalConvolution(module.weight, module.bias, module.stride, module.padding, module.dilation))
        else:
            # If the module is a container, recursively replace its layers
            modifyModel(module)

resnet18 = models.resnet18(pretrained=True)
modifyModel(resnet18)