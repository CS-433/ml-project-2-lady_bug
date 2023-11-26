import torch
from torch import nn


class FCN(torch.nn.Module):
    """Defines a fully connected network"""

    def __init__(self, module_dims, activation=nn.ReLU):
        """
        Construct FCN,
        module_dims is a list with input dimension, hidden dimensions and output dimension
        activation specifies the activation function
        """
        super().__init__()
        modules = []
        for i in range(len(module_dims) - 1):
            modules.append(nn.Linear(module_dims[i], module_dims[i + 1]))
            if i < len(module_dims) - 2:
                modules.append(activation())

        self.fcn = nn.Sequential(*modules)

    def forward(self, x):
        x = self.fcn(x)
        return x
