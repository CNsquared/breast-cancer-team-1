import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from src.models.autoencoder import GeneExpressionAutoencoder


class GeneExpressionRunner:
    def __init__(self, input_data: np.ndarray, latent_dim: int = 5, device: str = 'cpu'):
        self.X: np.ndarray = input_data  # shape (n_samples, n_genes)
        self.latent_dim: int = latent_dim
        self.device: str = device

    def build_model(self) -> nn.Module:
        model = GeneExpressionAutoencoder(
            input_dim=self.X.shape[1],
            latent_dim=self.latent_dim
        ).to(self.device)
        return model

    def train_autoencoder(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        X_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 16
    ) -> Tuple[nn.Module, float]:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            model.train()
            for batch in loader:
                x_batch = batch[0]
                optimizer.zero_grad()
                loss = criterion(model(x_batch), x_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_tensor), X_val_tensor).item()

        return model, val_loss

    def cross_validate(self, k: int = 5, epochs: int = 100) -> List[float]:
        scaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(self.X)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_losses: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            model = self.build_model()
            _, val_loss = self.train_autoencoder(model, X_train, X_val, epochs=epochs)
            fold_losses.append(val_loss)
            print(f"Fold {fold+1}/{k} - Val Loss: {val_loss:.4f}")

        return fold_losses

    def train_all_and_encode(self, epochs: int = 100) -> np.ndarray:
        scaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(self.X)

        model = self.build_model()
        model, _ = self.train_autoencoder(model, X_scaled, X_scaled, epochs=epochs)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        model.eval()
        with torch.no_grad():
            latent: np.ndarray = model.encode(X_tensor).cpu().numpy()

        return latent
