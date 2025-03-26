import torch
import torch.nn as nn
import torch.optim as optim
import pdb


class AutoencoderWithLogisticRegression(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoencoderWithLogisticRegression, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8000),
            nn.ReLU(),
            nn.Linear(8000, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8000),
            nn.ReLU(),
            nn.Linear(8000, input_dim)
        )
        
        # Logistic regression layer for response variable
        self.classifier = nn.Linear(latent_dim, 1)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        y_pred = torch.sigmoid(self.classifier(z))
        return x_recon, y_pred
