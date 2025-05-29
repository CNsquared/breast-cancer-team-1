import torch
import torch.nn as nn
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
import numpy as np
import copy
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from src.models.autoencoder import GeneExpressionAutoencoder
import random
import pandas as pd

class GeneExpressionRunner:
    def __init__(self, train_data: np.ndarray, latent_dim: int = 5, device: str = None, hidden_dims: List[int] = [128, 64], lr: float = 5e-4, batch_size: int = 16, dropout_rate: float = 0.2, weight_decay: float = 1e-5) -> None:
        self.X_train: np.ndarray = train_data  # shape (n_samples, n_genes)
        self.latent_dim: int = latent_dim
        self.device: str = device if device is not None else (
    "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
    )
)
        print(f"Using device: {self.device}")
        self.hidden_dims: List[int] = hidden_dims
        self.lr: float = lr  # learning rate
        self.batch_size: int = batch_size
        self.dropout_rate: float = dropout_rate  # Dropout rate for the autoencoder
        self.weight_decay: float = weight_decay  # Weight decay for the optimizer

    def build_model(self) -> nn.Module:
        model = GeneExpressionAutoencoder(
            input_dim=self.X_train.shape[1],
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        return model

    def train_autoencoder(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        X_val: np.ndarray,
        patience: int = 10,
        max_epochs: int = 200,
        delta: float = 1e-4,
    ) -> Tuple[nn.Module, float]:

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_model_state = None
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
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

            if val_loss < best_val_loss - delta:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        return model, best_val_loss

    def cross_validate(self, k: int = 5, patience: int = 10, max_epochs: int = 200, delta: float = 1e-4) -> List[float]:
        print("Performing cross-validation using training data")
        scaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(self.X_train)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_losses: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            model = self.build_model()
            _, val_loss = self.train_autoencoder(model, X_train, X_val, patience=patience, max_epochs=max_epochs, delta=delta)
            fold_losses.append(val_loss)
            print(f"Fold {fold + 1}/{k} - Val Loss: {val_loss:.4f}")

        return fold_losses

    def train_all_and_encode(self, patience: int = 10, max_epochs: int = 200, delta: float = 1e-4, return_model: bool = False) -> np.ndarray:
        print("Training autoencoder on all training data, returning latent representations")
        scaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(self.X_train)

        model = self.build_model()
        model, _ = self.train_autoencoder(model, X_scaled, X_scaled, patience=patience, max_epochs=max_epochs, delta=delta)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        model.eval()
        with torch.no_grad():
            latent: np.ndarray = model.encode(X_tensor).cpu().numpy()

        if return_model:
            return model, scaler
        else:
            return latent

    def trained_model_encode(self, trained_model: nn.Module, data: pd.DataFrame, fit_scaler: StandardScaler) -> np.ndarray:
        """
        Encodes the input data using the trained autoencoder model.
        """
        X = data.to_numpy()
        X_scaled = fit_scaler.transform(X)
        
        device = next(trained_model.parameters()).device

        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        trained_model.eval()
        with torch.no_grad():
            latent = trained_model.encode(X_scaled_tensor).cpu().numpy()

        return latent
        
    def get_weights(self, trained_model : nn.Module):
        layer = {}
        for name, param in trained_model.named_parameters():
            layer[name] = param
        return layer
