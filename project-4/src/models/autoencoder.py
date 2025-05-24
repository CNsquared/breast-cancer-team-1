import torch
import torch.nn as nn
from typing import List, Optional

class GeneExpressionAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 5, hidden_dims: Optional[List[int]] = None) -> None:

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        super().__init__()
        dims = [input_dim] + hidden_dims + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        # Decoder is symmetric
        rev_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        layers = []
        for i in range(len(rev_dims) - 1):
            layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
            if i < len(rev_dims) - 2:
                layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)