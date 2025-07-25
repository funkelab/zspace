import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils as tvutils

from tqdm import tqdm
import pathlib

from zspace.dataset import get_train_loader, get_valid_loader
from zspace.tarflow_model import get_tarflow_model
from config import UniversalConfig as ucon, TARFlowConfig as tcon

def compute_loss(tarflow_model, x, y):
    z, outputs, logdets = tarflow_model(x, y)
    loss = tarflow_model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)

if __name__ == "__main__":
    train_loader = get_train_loader(ucon)
    valid_loader = get_valid_loader(ucon)

    fixed_noise = torch.randn(tcon.num_classes * 10, (tcon.img_size // tcon.patch_size)**2, tcon.channel_size * tcon.patch_size ** 2, device=ucon.device)
    fixed_y = torch.arange(tcon.num_classes, device=ucon.device).view(-1, 1).repeat(1, 10).flatten()

    train_losses = []
    valid_losses = []

    ckpt_dir = pathlib.Path(tcon.save_dir_p_xy) / "checkpoints"
    sample_dir = pathlib.Path(tcon.save_dir_p_xy) / "sample_p_xy"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    sample_dir.mkdir(exist_ok=True, parents=True)    

    tarflow_model = get_tarflow_model(ucon, tcon)

    optimizer = torch.optim.AdamW(tarflow_model.parameters(), betas=(0.9, 0.95), lr=tcon.lr, weight_decay=tcon.weight_decay)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=tcon.num_epochs, eta_min=1e-6)

    for epoch in tqdm(range(tcon.num_epochs)):
        total_train_loss = 0
        total_valid_loss = 0

        tarflow_model.train()

        for batch_idx, batch in enumerate(train_loader):
            xc = batch[2].unsqueeze(1).to(ucon.device)
            y = batch[3].to(ucon.device)      

            eps = tcon.noise_std * torch.randn_like(xc)
            xc = xc + eps

            optimizer.zero_grad()

            loss, (z_t, outputs, logdets) = compute_loss(tarflow_model, xc, y)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            current_lr = lr_schedule.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * ucon.batch_size) )

        if epoch % tcon.sample_freq == 0 or epoch == 99:
            with torch.no_grad():
                xc_samples = tarflow_model.reverse(fixed_noise, fixed_y)

            tvutils.save_image(xc_samples, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
            tvutils.save_image(tarflow_model.unpatchify(z_t[:100]), sample_dir / f'uspace_{epoch:03d}.png', normalize=True, nrow=10)
            
            print('sampling complete')

        tarflow_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                xc = batch[2].unsqueeze(1).to(ucon.device)
                y = batch[3].to(ucon.device)

                eps = tcon.noise_std * torch.randn_like(xc)
                xc = xc + eps

                loss, (z_t, outputs, logdets) = compute_loss(tarflow_model, xc, y)
                total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * ucon.batch_size) )

        tqdm.write(f"\tepoch {epoch + 1} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

        if epoch % tcon.sample_freq == 0 or epoch == 99:
            torch.save(tarflow_model.state_dict(), ckpt_dir / f"model_epoch_{epoch}.pth")
            print(f"model saved at epoch {epoch}")

    print("training complete")

