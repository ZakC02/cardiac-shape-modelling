import torch
import torch.nn as nn
import torch.nn.functional as F


"""

This file contains everything about the VAE model used in the project:

- latent_dim
- Encoder
- Decoder
- VAE
- Loss function

- save_model function
- load_model function

"""

# Hyperparameters
latent_dim = 32
input_channels = 4  # Now we have 4 channels for the one-hot encoded images

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            self.conv_block(input_channels, 48),
            self.conv_block(48, 96),
            self.conv_block(96, 192),
            self.conv_block(192, 384)
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(384 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(384 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ELU()
        )

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Decoder, self).__init__()

        self.fc_decode = nn.Linear(latent_dim, 384 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            self.deconv_block(384, 192),
            self.deconv_block(192, 96),
            self.deconv_block(96, 48),
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 4, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        x = self.fc_decode(z)
        x = F.elu(x)
        x = x.view(-1, 384, 8, 8)
        x = self.decoder_conv(x)
        return x

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ELU()
        )

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x, testing=False):
        self.test = testing
        if not self.test:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decoder(z)
            return recon_x, mu, logvar
        else:
            mu, logvar = self.encoder(x)
            recon_x = self.decoder(mu)
            return recon_x, mu, logvar
        
    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

def loss_function(recon_x, x, mu, logvar, beta):
    # Calculate cross-entropy loss
    CE = F.cross_entropy(recon_x, x, reduction='sum') / x.size(0)

    # Calculate KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis = -1).mean()

    # Return total loss
    return CE + beta * KLD
    # return CE


def save_model(model, path):
    """
    Saves the VAE model to a file.

    Args:
    - model (nn.Module): The VAE model to save.
    - path (str): The file path to save the model.

    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    Loads the VAE model from a file.

    Args:
    - model (nn.Module): The VAE model instance.
    - path (str): The file path to load the model from.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")