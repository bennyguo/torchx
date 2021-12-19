import torch.nn as nn
import models


@models.register('mlp')
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim, self.out_dim = config.model.in_dim, config.model.out_dim
        self.hidden_dims = config.model.hidden_dims
        layers = []
        lastd = self.in_dim
        for hidden in self.hidden_dims:
            layers.append(nn.Linear(lastd, hidden))
            layers.append(nn.ReLU())
            lastd = hidden
        layers.append(nn.Linear(lastd, self.out_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.layers(x)
        return x
