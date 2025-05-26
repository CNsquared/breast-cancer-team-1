import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Variational Autoencoder (no clustering loss)
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=None, beta=1.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [600, 800, 1000, 1200]
        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1), nn.Dropout(0.2)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        # Decoder
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1), nn.Dropout(0.2)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)
        self.beta = beta

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss(self, x, x_recon, mu, logvar):
        recon = nn.MSELoss(reduction='mean')(x_recon, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kld, recon, kld

class SamplingRunner:
    def __init__(self, latent_dim=1500, sample_size=1000,
                 hidden_dims=None, beta=1.0,
                 pretrain_epochs=300, batch_size=32, lr=5e-3,
                 k_range=range(2, 11)):
        self.latent_dim = latent_dim
        self.sample_size = sample_size
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k_range = k_range

    def run(self, df, verbose=False):
        # Prepare data
        X = df.values if isinstance(df, pd.DataFrame) else df
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        data = torch.tensor(X_scaled, dtype=torch.float32)
        loader = DataLoader(TensorDataset(data), batch_size=self.batch_size, shuffle=True)

        # Model setup
        model = VariationalAutoencoder(
            input_dim=X.shape[1], latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims, beta=self.beta
        )
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

        # Pre-train VAE
        if verbose: print(f"[INFO] Training VAE for {self.pretrain_epochs} epochs...")
        for epoch in range(self.pretrain_epochs):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                x_recon, mu, logvar, _ = model(batch)
                loss, recon, kld = model.loss(batch, x_recon, mu, logvar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(batch)
            scheduler.step(total_loss / len(X))
            if verbose and (epoch+1) % max(1, self.pretrain_epochs//5) == 0:
                print(f"Epoch {epoch+1}/{self.pretrain_epochs}, Loss: {total_loss/len(X):.4f}")

        # Encode all data
        with torch.no_grad():
            _, mu_all, _, z_all = model(data)
            Z = mu_all.cpu().numpy()

        # Sweep over k to find best silhouette
        best_k, best_score = None, -1
        best_labels = None
        if verbose: print("[INFO] Sweeping K for stable clusters...")
        for k in self.k_range:
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(Z)
            score = silhouette_score(Z, labels)
            if verbose: print(f"  k={k}, silhouette={score:.4f}")
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
        if verbose: print(f"[INFO] Best k={best_k} with silhouette={best_score:.4f}")

        # Compute cluster centers in latent space
        centers = np.vstack([
            Z[best_labels == i].mean(axis=0) for i in range(best_k)
        ])

        # Sample new points around centers
        per = self.sample_size // best_k
        Z_new = np.vstack([
            np.random.multivariate_normal(mu, np.eye(self.latent_dim)*1e-2, per)
            for mu in centers
        ])
        with torch.no_grad():
            X_new_scaled = model.decode(torch.tensor(Z_new, dtype=torch.float32)).cpu().numpy()
        X_new = scaler.inverse_transform(X_new_scaled)
        X_aug = np.vstack([X, X_new])

        if verbose: print("[INFO] Augmentation complete.")
        return pd.DataFrame(X_aug, columns=df.columns)

# Example usage:
# runner = SamplingRunnerVAE(latent_dim=1500, sample_size=2000)
# df_aug, best_k, sil = runner.run(df, verbose=True)
