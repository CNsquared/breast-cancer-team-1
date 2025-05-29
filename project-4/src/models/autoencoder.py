import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
from typing import List, Optional

class GeneExpressionAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2  # Dropout rate between 0 and 1
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Encoder
        dims = [input_dim] + hidden_dims + [latent_dim]
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.Dropout(dropout_rate))
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (symmetric)
        rev_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(rev_dims) - 1):
            decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
            if i < len(rev_dims) - 2:
                decoder_layers.append(nn.Dropout(dropout_rate))
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
