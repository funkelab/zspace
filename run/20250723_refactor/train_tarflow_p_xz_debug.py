import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from zspace.dataset import get_train_loader, get_valid_loader
from zspace.tarflow_model import get_tar_model
from zspace.vae_model import get_vae


from config import UniversalConfig as ucon, TARFlowConfig as tcon, VAEConfig as vcon

def compute_loss(tar_model, x, y):
    z, outputs, logdets = tar_model(x, y)
    loss = tar_model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)

if __name__ == "__main__":
    train_loader = get_train_loader(ucon)
    valid_loader = get_valid_loader(ucon)

    vae_model = get_vae(ucon, vcon, load_weights=True)

    tar_model = get_tar_model(ucon, tcon)

    optimizer = torch.optim.AdamW(tar_model.parameters(), betas=(0.9, 0.95), lr=tcon.lr, weight_decay=tcon.weight_decay)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=tcon.num_epochs, eta_min=1e-6)

    train_losses = []
    valid_losses = []

    total_train_loss = 0
    total_valid_loss = 0

    tar_model.train()

    for batch_idx, batch in enumerate(train_loader):
        xb = batch[1]
        xc = batch[2]
        y = batch[3]

        xb = torch.flatten(xb, start_dim=1, end_dim=-1).to(ucon.device)
        xc = xc.unsqueeze(1).to(ucon.device)
        y = y.to(ucon.device)

        with torch.no_grad():
            mean, logvar = vae_model.encoders["xb"](xb)
            zb = vae_model.reparameterize(mean, logvar)
        
        eps = tcon.noise_std * torch.randn_like(xc)
        xc = xc + eps

        optimizer.zero_grad()
        print(f"{zb.dtype=}")
        loss, (z_t, outputs, logdets) = compute_loss(tar_model, xc, zb)
        total_train_loss += loss.item()

    print("training complete")