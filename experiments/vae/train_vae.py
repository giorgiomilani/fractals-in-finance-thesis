import torch, torch.nn as nn
from fractalfinance.gaf.dataset import GAFWindowDataset
import numpy as np, typer, rich

class VAE(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128, 256), nn.ReLU(),
            nn.Linear(256, latent*2)   # mean & logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 128*128), nn.Sigmoid()
        )
    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5*logvar)
        recon = self.decoder(z).view(-1, 1,128,128)
        return recon, mu, logvar

@app.command()
def train(data_path: str):
    series = np.load(data_path)
    ds = GAFWindowDataset(series, resize=128)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    vae = VAE()
    opt = torch.optim.Adam(vae.parameters(), 1e-3)
    for ep in range(3):
        for x,_ in dl:
            recon, mu, logvar = vae(x)
            kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
            recon_loss = ((x - recon)**2).mean()
            loss = recon_loss + 1e-3*kld
            opt.zero_grad(); loss.backward(); opt.step()
        rich.print(ep, loss.item())
