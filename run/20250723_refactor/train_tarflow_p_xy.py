import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils as tvutils

from tqdm import tqdm
import pathlib

from zspace.dataset import get_train_loader, get_valid_loader
from zspace.tar_model import Model

from config import UniversalConfig as ucon, TARFlowConfig as tcon, ExperimentConfig as econ

def compute_loss(tar_model, x, y):
    z, outputs, logdets = tar_model(x, y)
    loss = tar_model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)

if __name__ == "__main__":
    train_loader = get_train_loader(ucon)
    valid_loader = get_valid_loader(ucon)

    fixed_noise = torch.randn(tcon.num_classes * 10, (tcon.img_size // tcon.patch_size)**2, tcon.channel_size * tcon.patch_size ** 2, device=tcon.device)
    fixed_y = torch.arange(tcon.num_classes, device=tcon.device).view(-1, 1).repeat(1, 10).flatten()

    train_losses = []
    valid_losses = []

    model_name = f'tar_{tcon.patch_size}_{tcon.channels}_{tcon.num_blocks}_{tcon.layers_per_block}_{tcon.noise_std:.2f}'
    output_path = pathlib.Path(econ.experiment_outputs)
    ckpt_file = output_path / f'xc_model_{model_name}.pth'
    sample_dir = output_path / f'xc_samples_{model_name}'
    sample_dir.mkdir(exist_ok=True, parents=True)    

    tar_model = Model(
        in_channels=tcon.in_channels,
        img_size=tcon.img_size,
        patch_size=tcon.patch_size,
        channels=tcon.channels,
        num_blocks=tcon.num_blocks,
        layers_per_block=tcon.layers_per_block,
        nvp=tcon.nvp,
        num_classes=tcon.num_classes
    )

    tar_model = tar_model.to(tcon.device)
    optimizer = torch.optim.AdamW(tar_model.parameters(), betas=(0.9, 0.95), lr=tcon.lr, weight_decay=tcon.weight_decay)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=tcon.num_epochs, eta_min=1e-6)

    for epoch in tqdm(range(tcon.num_epochs)):
        total_train_loss = 0
        total_valid_loss = 0

        tar_model.train()

        for batch_idx, batch in enumerate(train_loader):
            xc = batch[2].unsqueeze(1).to(tcon.device)
            y = batch[3].to(tcon.device)      

            eps = tcon.noise_std * torch.randn_like(xc)
            xc = xc + eps

            optimizer.zero_grad()

            loss, (z_t, outputs, logdets) = compute_loss(xc, y)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            current_lr = lr_schedule.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * ucon.batch_size) )

        if (epoch + 1) % tcon.sample_freq == 0:
            with torch.no_grad():
                xc_samples = tar_model.reverse(fixed_noise, fixed_y)

            tvutils.save_image(xc_samples, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
            tvutils.save_image(tar_model.unpatchify(z_t[:100]), sample_dir / f'uspace_{epoch:03d}.png', normalize=True, nrow=10)
            
            print('sampling complete')

        tar_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                xc = batch[2].unsqueeze(1).to(tcon.device)
                y = batch[3].to(tcon.device)

                eps = tcon.noise_std * torch.randn_like(xc)
                xc = xc + eps

                loss, (z_t, outputs, logdets) = compute_loss(xc, y)
                total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * ucon.batch_size) )

        tqdm.write(f"\tepoch {epoch + 1} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

    print("training complete")

    torch.save(tar_model.state_dict(), ckpt_file)


