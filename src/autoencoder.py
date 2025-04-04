import torch
import torch.nn as nn
import torch.optim as optim
import pdb


class AutoencoderWithLogisticRegression(nn.Module):
    def __init__(self, input_dim, latent_dim, objective, tradeoff = 0.5):
        super(AutoencoderWithLogisticRegression, self).__init__()

        self.tradeoff = tradeoff

        if objective == 'classification':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif objective == 'regression':
            self.criterion = nn.MSELoss(reduction='mean')
        
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
    
    def reconstruction_loss(self, x, x_recon):
        return nn.functional.mse_loss(x, x_recon)
    
    def task_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)
    
    def total_loss(self, x, x_recon, y_pred, y_true):
        return self.tradeoff * self.reconstruction_loss(x, x_recon) + (1 - self.tradeoff) * self.task_loss(y_pred, y_true)
