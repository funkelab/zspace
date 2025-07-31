import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils as tvutils

from tqdm import tqdm
import pathlib

from zspace.dataset import get_valid_dataset, get_train_loader, get_valid_loader
from zspace.tarflow_model import get_tarflow_model
from zspace.vae_model import get_vae
from config import UniversalConfig as ucon, TARFlowConfig as tcon, VAEConfig as vcon
from train_tarflow_p_xz import compute_loss

def sample_debug(tarflow_model, save_dir, epoch=None):
    sample_dir = save_dir / "sample_p_xz"
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    batch_size = 1
    num_tokens = (tcon.img_size // tcon.patch_size)**2
    token_size = tcon.channel_size * tcon.patch_size ** 2
    # z shape : (batch_size, 1, img_size, img_size)
    z = torch.randn(batch_size, 1,, dtype=torch.float32)
    y_0 = torch.zeros((batch_size, tcon.cond_dim), dtype=torch.float32)
    y_1 = torch.ones((batch_size, tcon.cond_dim), dtype=torch.float32)

    x_0 = tarflow_model.reverse(z, y_0)
    x_1 = tarflow_model.reverse(z, y_1)
    xs = [x_0, x_1]
    print(f"{x_0=}, {x_1=}")

    if epoch is not None:
        save_dir = sample_dir / f"y_0_epoch_{epoch}.png"
    else:
        save_dir = sample_dir / "y_0.png"

    tvutils.save_image(xs, save_dir, normalize=True, nrow=1)
    del xs
    torch.cuda.empty_cache()

if __name__ == "__main__":
    valid_dataset = get_valid_dataset(ucon)

    train_loader = get_train_loader(ucon)
    valid_loader = get_valid_loader(ucon)

    vae_model = get_vae(ucon, vcon, ckpt_file=vcon.ckpt_file)
    tarflow_model = get_tarflow_model(ucon, tcon)

    optimizer = torch.optim.AdamW(tarflow_model.parameters(), betas=(0.9, 0.95), lr=tcon.lr, weight_decay=tcon.weight_decay)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=tcon.num_epochs, eta_min=1e-6)

    save_dir = pathlib.Path(tcon.save_dir_p_xz)
    ckpt_dir = save_dir / "checkpoints"
    sample_dir = save_dir / "sample_p_xz_debug"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    sample_dir.mkdir(exist_ok=True, parents=True)  

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

            # with torch.no_grad():
            #     zb, logvar = vae_model.encoders["xb"](xb)   # no reparameterization, deterministic encoding
            
            zb = (torch.ones((ucon.z_dim, ucon.batch_size), dtype=torch.float32) * y).T

            eps = tcon.noise_std * torch.randn_like(xc)
            xc = xc + eps

            optimizer.zero_grad()

            loss, (z_t, outputs, logdets) = compute_loss(tarflow_model, xc, zb)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            current_lr = lr_schedule.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * ucon.batch_size) )

        if epoch % tcon.sample_freq == 0 or epoch == 99:
            sample_debug(tarflow_model, sample_dir, epoch=epoch)
            print("sampling complete")
            torch.save(tarflow_model.state_dict(), ckpt_dir / f"tarflow_model_debug_epoch_{epoch}.pth")
            print(f"model saved at epoch {epoch}")