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
    

class FCN_with_last_activation(torch.nn.Module):
    """Defines a fully connected network with non-linearity applied after the model output"""

    def __init__(self, module_dims, activation=nn.ELU, last_activation=nn.Softsign):
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
            elif i == len(module_dims) - 2:
                modules.append(last_activation())

        self.fcn = nn.Sequential(*modules)

    def forward(self, x):
        x = self.fcn(x)
        return x


class FCN_with_parallel_branch(torch.nn.Module):
    """ Defines a fully connected network with timesteps and physics parameters processed separately in parallel branches"""
    def __init__(self, timestep_module_dims, param_module_dims, common_module_dim, activation=nn.ELU, last_activation=nn.Softsign):
        """
        Construct FCN,
        module_dims is a list with input dimension, hidden dimensions and output dimension for timestep branch
        activation specifies the activation function
        """
        super().__init__()
        
        assert(timestep_module_dims[0] == 1) # specifies input dimension for timestep branch
        assert(timestep_module_dims[-1] == param_module_dims[-1])
        assert(timestep_module_dims[-1] == common_module_dim[0])

        self.timestep_branch = nn.Sequential(*self.get_modules(timestep_module_dims, activation))
        # branch for processing physics parameters is fixed
        self.param_branch = nn.Sequential(*self.get_modules(param_module_dims, activation))

        # branch for combination
        self.combine_branch = nn.Sequential(*self.get_modules(common_module_dim, activation))

        self.last_activation = last_activation()

    def forward(self, x):
        timestep_output = self.timestep_branch(x[:, 0].reshape(-1, 1))
        param_output = self.param_branch(x[:, 1:])
        output = self.combine_branch(timestep_output + param_output)
        return self.last_activation(output)
    
    def get_modules(self, module_dims, activation):
        modules = []
        for i in range(len(module_dims) - 1):
            modules.append(nn.Linear(module_dims[i], module_dims[i + 1]))
            if i < len(module_dims) - 2:
                modules.append(activation())
        return modules