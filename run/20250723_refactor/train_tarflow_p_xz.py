import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils as tvutils

from tqdm import tqdm
import pathlib
import matplotlib.pyplot as plt

from zspace.dataset import get_valid_dataset, get_train_loader, get_valid_loader
from zspace.tarflow_model import get_tarflow_model
from zspace.vae_model import get_vae
from config import UniversalConfig as ucon, TARFlowConfig as tcon, VAEConfig as vcon

def compute_loss(tarflow_model, x, y):
    z, outputs, logdets = tarflow_model(x, y)
    loss = tarflow_model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)

def sample(dataset, vae_model, tarflow_model, save_dir, epoch=None):
    sample_dir = save_dir / "sample_p_xz"
    sample_dir.mkdir(exist_ok=True, parents=True)

    num_zbs = 2
    zbs_0 = []
    zbs_1 = []

    for sample in dataset:
        if len(zbs_0) == num_zbs and len(zbs_1) == num_zbs:
            break
        
        _, xb, _, y = sample
        with torch.no_grad():
            mean, logvar = vae_model.encoders["xb"](xb)
            zb = vae_model.reparameterize(mean, logvar)

        if y == 0 and len(zbs_0) < num_zbs:
            zbs_0.append(zb)
        elif y == 1 and len(zbs_1) < num_zbs:
            zbs_1.append(zb)

    num_samples_per_class = 10
    fixed_noise = torch.randn(
        tcon.num_classes * num_samples_per_class, 
        (tcon.img_size // tcon.patch_size)**2, 
        tcon.channel_size * tcon.patch_size ** 2, 
        device=ucon.device)

    for i, zb in enumerate(zbs_0):
        # print("reconstructing xcs with y=0")
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(fixed_noise, zb)

        if epoch is not None:
            save_dir = sample_dir / f"y_0_epoch_{epoch}_sample_{i}.png"
        else:
            save_dir = sample_dir / f"y_0_sample_{i}.png"

        tvutils.save_image(xc_samples, save_dir, normalize=True, nrow=num_samples_per_class)
        del xc_samples
        torch.cuda.empty_cache()

    for i, zb in enumerate(zbs_1):
        # print("reconstructing xcs with y=1")
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(fixed_noise, zb)

        if epoch:
            save_dir = sample_dir / f"y_1_epoch_{epoch}_sample_{i}.png"
        else:
            save_dir = sample_dir / f"y_1_sample_{i}.png"
        tvutils.save_image(xc_samples, save_dir, normalize=True, nrow=num_samples_per_class)

        del xc_samples
        torch.cuda.empty_cache()

def plot_losses(train_losses, valid_losses, save_dir):
    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "losses.png")
    plt.close()

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
    sample_dir = save_dir / "sample_p_xz"
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

        if epoch % tcon.sample_freq == 0 or epoch == 99:
            sample(valid_dataset, vae_model, tarflow_model, save_dir, epoch=epoch)
            print("sampling complete")

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
                    mean, logvar = vae_model.encoders["xb"](xb)
                    zb = vae_model.reparameterize(mean, logvar)
                
                eps = tcon.noise_std * torch.randn_like(xc)
                xc = xc + eps

                loss, (z_t, outputs, logdets) = compute_loss(tarflow_model, xc, zb)
                total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * ucon.batch_size) )

        tqdm.write(f"\tepoch {epoch + 1} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

        if epoch % tcon.sample_freq == 0 or epoch == 99:
            torch.save(tarflow_model.state_dict(), ckpt_dir / f"tarflow_model_epoch_{epoch}.pth")
            print(f"model saved at epoch {epoch}")

    print("training complete")

    plot_losses(train_losses, valid_losses, save_dir)