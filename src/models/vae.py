"""vae.py module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super(VAE, self).__init__()
        
        # Encoder
        modules = []
        in_features = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(in_features, h_dim))
            modules.append(nn.ReLU())
            in_features = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        modules = []
        hidden_dims.reverse()
        in_features = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(in_features, h_dim))
            modules.append(nn.ReLU())
            in_features = h_dim
        modules.append(nn.Linear(hidden_dims[-1], input_dim))
        self.decoder = nn.Sequential(*modules)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
        
    def decode(self, z):
        return self.decoder(z)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

