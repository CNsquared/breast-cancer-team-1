import torch
import torch.nn as nn

class GeneExpressionAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 500, latent_dim: int = 5) -> None:
        super(GeneExpressionAutoencoder, self).__init__()

        # Encoder: compress to latent space
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)  # Final latent layer
        )

        # Decoder: reconstruct input
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # Final output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
