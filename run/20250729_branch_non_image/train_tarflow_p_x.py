import config as c
import utils as u
from vae_model import get_vae
from tarflow_model import get_tarflow_model
from zspace.dataset import get_train_dataset, get_valid_dataset, get_train_loader, get_valid_loader

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

if __name__ == "__main__":
    vae_model = get_vae(c, ckpt_file=u.get_ckpt_file(c.vae_model_name))

    input_dim = c.input_dims["xc"]   # learning p(xc|y)
    tarflow_model = get_tarflow_model(c, input_dim, num_classes=c.num_classes)

    train_dataset = get_train_dataset(c)
    valid_dataset = get_valid_dataset(c)

    train_loader = get_train_loader(c)
    valid_loader = get_valid_loader(c)

    sample_dir = u.get_save_dir(c.tarflow_p_x_model_name) / "xc_samples"
    ckpt_dir = u.get_save_dir(c.tarflow_p_x_model_name) / "checkpoints"
    sample_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    optimizer = torch.optim.AdamW(tarflow_model.parameters(), betas=(0.9, 0.95), lr=c.lr, weight_decay=c.weight_decay)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=c.num_epochs_tarflow, eta_min=1e-6)

    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(c.num_epochs_tarflow)):
        total_train_loss = 0
        total_valid_loss = 0

        tarflow_model.train()

        for batch_idx, batch in enumerate(train_loader):
                xc = batch[2]                        # shape : (batch_size, input_dims["xb"])

                xc = xc.view(xc.size(0), input_dim, c.token_size).to(c.device)   # shape : (batch_size, num_tokens, token_size)

                eps = c.noise_std * torch.randn_like(xc)
                xc = xc + eps
                xc = xc.to(c.device)

                optimizer.zero_grad()

                z, outputs, logdets = tarflow_model(xc)
                loss = tarflow_model.get_loss(z, logdets)

                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                current_lr = lr_schedule.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * c.batch_size) )

        tarflow_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                    xc = batch[2]                        # shape : (batch_size, input_dims["xc"])

                    xc = xc.view(xc.size(0), input_dim, c.token_size)   # shape : (batch_size, num_tokens, token_size)

                    eps = c.noise_std * torch.randn_like(xc)
                    xc = xc + eps
                    xc = xc.to(c.device)

                    z, outputs, logdets = tarflow_model(xc)
                    loss = tarflow_model.get_loss(z, logdets)

                    total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * c.batch_size) )

        tqdm.write(f"\tepoch {epoch} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

        if (epoch + 1) % c.sample_freq == 0 or epoch == 0 or epoch == 99:
            u.sample_p_x(tarflow_model, sample_dir, epoch=epoch)
            torch.save(tarflow_model.state_dict(), ckpt_dir / f"{c.tarflow_p_x_model_name}_epoch_{epoch}.pth")
            tqdm.write(f"saved and sampled at epoch {epoch + 1}")

    print("training complete")
    u.plot_losses(train_losses, valid_losses, u.get_save_dir(c.tarflow_p_x_model_name))