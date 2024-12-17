"""GAN implementation for financial data generation."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

class Generator(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, feature_dim))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class FinancialGAN:
    def __init__(self, config: Dict):
        """Initialize the GAN."""
        self.latent_dim = config['model']['latent_dim']
        self.feature_dim = config['model']['feature_dim']
        self.hidden_dims = config['model']['hidden_dims']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.generator = Generator(
            self.latent_dim, 
            self.feature_dim, 
            self.hidden_dims
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.feature_dim, 
            self.hidden_dims[::-1]
        ).to(self.device)
        
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=config['model']['learning_rate']
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config['model']['learning_rate']
        )
        
        self.criterion = nn.BCELoss()
        
    def train_step(self, real_data: torch.Tensor) -> Tuple[float, float]:
        """Perform one training step."""
        batch_size = real_data.size(0)
        real_label = torch.ones(batch_size, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        d_real_output = self.discriminator(real_data)
        d_real_loss = self.criterion(d_real_output, real_label)
        
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        d_fake_output = self.discriminator(fake_data.detach())
        d_fake_loss = self.criterion(d_fake_output, fake_label)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        g_output = self.discriminator(fake_data)
        g_loss = self.criterion(g_output, real_label)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
        
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic financial data."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
        self.generator.train()
        return samples

