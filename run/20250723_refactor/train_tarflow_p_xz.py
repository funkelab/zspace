import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from zspace.dataset import get_train_loader, get_valid_loader
from zspace.tarflow_model import get_tarflow_model
from zspace.vae_model import get_vae
from config import UniversalConfig as ucon, TARFlowConfig as tcon, VAEConfig as vcon

def compute_loss(tarflow_model, x, y):
    z, outputs, logdets = tarflow_model(x, y)
    loss = tarflow_model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)

if __name__ == "__main__":
    train_loader = get_train_loader(ucon)
    valid_loader = get_valid_loader(ucon)

    vae_model = get_vae(ucon, vcon, ckpt_file=vcon.ckpt_file)
    tarflow_model = get_tarflow_model(ucon, tcon)

    optimizer = torch.optim.AdamW(tarflow_model.parameters(), betas=(0.9, 0.95), lr=tcon.lr, weight_decay=tcon.weight_decay)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=tcon.num_epochs, eta_min=1e-6)

    ckpt_file = tcon.ckpt_file_p_xz

    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(tcon.num_epochs)):
        total_train_loss = 0
        total_valid_loss = 0

        tarflow_model.train()

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

            loss, (z_t, outputs, logdets) = compute_loss(tarflow_model, xc, zb)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            current_lr = lr_schedule.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * ucon.batch_size) )

        tarflow_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                xb = batch[1]
                xc = batch[2]
                y = batch[3]

                xb = torch.flatten(xb, start_dim=1, end_dim=-1).to(ucon.device)
                xc = xc.unsqueeze(1).to(ucon.device)
                y = y.to(ucon.device)

                with torch.no_grad():
                    zb, logvar = vae_model.encoders["xb"](xb)
                    # zb = vae_model.reparameterize(zb, logvar)
                
                eps = tcon.noise_std * torch.randn_like(xc)
                xc = xc + eps

                loss, (z_t, outputs, logdets) = compute_loss(tarflow_model, xc, zb)
                total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * ucon.batch_size) )

        tqdm.write(f"\tepoch {epoch + 1} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

    print("training complete")

    torch.save(tarflow_model.state_dict(), ckpt_file)