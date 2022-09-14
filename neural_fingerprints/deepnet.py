
from torch import nn
import torch.nn.functional as F

class DeepNetwork(nn.Module):
    """Composes a fingerprint function with signature (smiles, weights, params)
     with a fully-connected neural network."""
    def __init__(self, layer_sizes,normalize=False):
        super(DeepNetwork, self).__init__()
        self.layer_sizes = layer_sizes + [1]
        self.normalize = normalize
        self.linears = nn.ModuleList()
        # build the model
        for (in_size,out_size) in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.linears.append(nn.Linear(in_size,out_size))
        
    def forward(self, x):
        for layer_index in range(len(self.layer_sizes) - 1):
            x = self.linears[layer_index](x)
            if layer_index < len(self.layer_sizes) - 2:
                if self.normalize:
                    x = F.normalize(x)
                x = F.relu(x)
        return x[:,0]

