import torch
import torch.nn as nn
import torch.nn.functional as F

# Small building blocks to reduce repetition and make the model clearer
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return F.gelu(self.block(x) + x)

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # downsample 3->16
        self.initial = ConvBlock(3, 16, stride=2)
        self.res1 = nn.Sequential(
            ResidualBlock(16), 
            ResidualBlock(16), 
            ResidualBlock(16), 
            ResidualBlock(16), 
        )
        # downsample 16->32
        self.down2 = ConvBlock(16, 32, stride=2)
        self.res2 = nn.Sequential(
            ResidualBlock(32), 
            ResidualBlock(32), 
            ResidualBlock(32), 
            ResidualBlock(32), 
        )
        # downsample 32->64
        self.down3 = ConvBlock(32, 64, stride=2)
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        x = self.down3(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.up1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.res1 = nn.Sequential(
            ResidualBlock(32), 
            ResidualBlock(32), 
            ResidualBlock(32), 
            ResidualBlock(32), 
        )
        self.up2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.res2 = nn.Sequential(
            ResidualBlock(16), 
            ResidualBlock(16), 
            ResidualBlock(16), 
            ResidualBlock(16), 
        )
        self.up3 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn16 = nn.BatchNorm2d(16)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 64, 4, 4)
        x = F.gelu(self.bn32(self.up1(x)))
        x = self.res1(x)
        x = F.gelu(self.bn16(self.up2(x)))
        x = self.res2(x)
        x = torch.sigmoid(self.up3(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    
class LatentNormVAE(VAE):
    def __init__(self, latent_dim=128):
        super().__init__(latent_dim=latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = self.layer_norm(z)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
