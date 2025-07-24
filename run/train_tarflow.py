# %%
import numpy as np

# %%
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils as tvutils

from tqdm import tqdm
import pathlib

from zspace.dataset import ToyModel
from zspace.tar_model import Model
from zspace.vae_model import Encoder, Decoder, JointVAE
from notebooks.config import UniversalConfig as ucon, VAEConfig as vcon, TARFlowConfig as tcon

# %%
# model hyperparameters

batch_size = 64
lr = 1e-4
train_seed = 42
valid_seed = 24

# %%
# load data

train_dataset = ToyModel()
valid_dataset = ToyModel()

train_gen = torch.Generator()
valid_gen = torch.Generator()

train_gen.manual_seed(ucon.train_seed)
valid_gen.manual_seed(ucon.valid_seed)

train_loader = DataLoader(dataset=train_dataset, batch_size=ucon.batch_size, generator=train_gen, shuffle=True, **ucon.kwargs)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=ucon.batch_size, generator=valid_gen, shuffle=False, **ucon.kwargs)


# %%
# define loss function

def compute_loss(x, y):
    z, outputs, logdets = tar_model(x, y)
    loss = tar_model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)

# %%
# train model to learn p(xc|y)

fixed_noise = torch.randn(tcon.num_classes * 10, (tcon.img_size // tcon.patch_size)**2, tcon.channel_size * tcon.patch_size ** 2, device=tcon.device)
fixed_y = torch.arange(tcon.num_classes, device=tcon.device).view(-1, 1).repeat(1, 10).flatten()

train_losses = []
valid_losses = []

model_name = f'tar_{tcon.patch_size}_{tcon.channels}_{tcon.num_blocks}_{tcon.layers_per_block}_{tcon.noise_std:.2f}'
output_path = pathlib.Path('notebook_outputs')
sample_dir = output_path / f'xc_samples_{model_name}'
ckpt_file = output_path / f'xc_model_{model_name}.pth'
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
optimizer = torch.optim.AdamW(tar_model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
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

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

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

    valid_losses.append( total_valid_loss / ((batch_idx + 1) * batch_size) )

    tqdm.write(f"\tepoch {epoch + 1} complete")
    tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
    tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

print("training complete")

torch.save(tar_model.state_dict(), ckpt_file)

# %%
# train model to learn p(xc|zb)

vae_model = JointVAE(encoders=vcon.encoders, decoders=vcon.decoders, device=ucon.device).to(ucon.device)
vae_model.load_state_dict(torch.load(vcon.ckpt_file))

fixed_noise = torch.randn(tcon.num_classes * 10, (tcon.img_size // tcon.patch_size)**2, tcon.channel_size * tcon.patch_size ** 2, device=ucon.device)
fixed_y = torch.arange(tcon.num_classes, device=ucon.device).view(-1, 1).repeat(1, 10).flatten()

zbs_0 = []
zbs_1 = []
zbs = []
for sample in valid_dataset:
    _, xb, _, y = sample
    zb = vcon.encoders["xb"](xb)

    if y == 0 and len(zbs_0) < 5:
        zbs_0.append(zb)
    elif y == 1 and len(zbs_1) < 5:
        zbs_1.append(zb)
    else:
        break

tar_model = Model(
    in_channels=tcon.in_channels,
    img_size=tcon.img_size,
    patch_size=tcon.patch_size,
    channels=tcon.channels,
    num_blocks=tcon.num_blocks,
    layers_per_block=tcon.layers_per_block,
    nvp=tcon.nvp,
    num_classes=tcon.num_classes,
    cond_dim=tcon.latent_dim
)

tar_model = tar_model.to(ucon.device)
optimizer = torch.optim.AdamW(tar_model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
lr_schedule = CosineAnnealingLR(optimizer, T_max=tcon.num_epochs, eta_min=1e-6)

train_losses = []
valid_losses = []

for epoch in tqdm(range(tcon.num_epochs)):
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

        loss, (z_t, outputs, logdets) = compute_loss(xc, zb)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        current_lr = lr_schedule.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

    if (epoch + 1) % tcon.sample_freq == 0:
        with torch.no_grad():
            for zb in zbs_0:
                xc_samples_0 = tar_model.reverse(fixed_noise, zb)
            for zb in zbs_1:
                xc_samples_1 = tar_model.reverse(fixed_noise, zb)

        tvutils.save_image(xc_samples_0, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
        tvutils.save_image(xc_samples_1, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
        tvutils.save_image(tar_model.unpatchify(z_t[:100]), sample_dir / f'uspace_{epoch:03d}.png', normalize=True, nrow=10)
        
        print('sampling complete')

    tar_model.eval()

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

            optimizer.zero_grad()

            # loss, (z_t, outputs, logdets) = compute_loss(xc, zb)
            loss, (z_t, outputs, logdets) = compute_loss(xc, y)
            total_valid_loss += loss.item()

    valid_losses.append( total_valid_loss / ((batch_idx + 1) * batch_size) )

    tqdm.write(f"\tepoch {epoch + 1} complete")
    tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
    tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

print("training complete")

torch.save(tar_model.state_dict(), ckpt_file)

# %%
# evaluate model as classifier
# note: only use this for p(xc | y) model

num_correct = 0
num_examples = 0

tar_model.eval()

for batch_idx, batch in enumerate(valid_loader):
    xc = batch[2]
    y = batch[3].to(ucon.device)

    xc = xc.unsqueeze(1).to(ucon.device)
    y = y.to(ucon.device)

    eps = tcon.noise_std * torch.randn_like(xc)
    xc = xc.repeat(tcon.num_classes, 1, 1, 1)
   
    y_ = torch.arange(tcon.num_classes, device=ucon.device).view(-1, 1).repeat(1, y.size(0)).flatten()

    with torch.no_grad():
        z, outputs, logdets = tar_model(xc, y_)
        losses = 0.5 * z.pow(2).mean(dim=[1, 2]) - logdets # keep the batch dimension
        pred = losses.reshape(tcon.num_classes, y.size(0)).argmin(dim=0)
    
    if batch_idx == 0:
        print(f"y shape: {y.shape}")
        print(f"Prediction: {pred}, true: {y}")

    num_correct += (pred == y).sum()
    num_examples += y.size(0)

print(f'accuracy: {100 * num_correct / num_examples:.2f}%')

# %%
# train model to learn p(zc|zb)

vae_model = JointVAE(encoders=vcon.encoders, decoders=vcon.decoders, device=ucon.device).to(ucon.device)
vae_model.load_state_dict(torch.load(vcon.ckpt_file))

fixed_noise = torch.randn(tcon.num_classes * 10, (tcon.img_size // tcon.patch_size)**2, tcon.channel_size * tcon.patch_size ** 2, device=ucon.device)
fixed_y = torch.arange(tcon.num_classes, device=ucon.device).view(-1, 1).repeat(1, 10).flatten()
fixed_zb = torch.randn(tcon.num_classes * 10, tcon.latent_dim, device=ucon.device)

tar_model = Model(
    in_channels=tcon.in_channels,
    img_size=tcon.img_size,
    patch_size=tcon.patch_size,
    channels=tcon.channels,
    num_blocks=tcon.num_blocks,
    layers_per_block=tcon.layers_per_block,
    nvp=tcon.nvp,
    num_classes=tcon.num_classes,
    cond_dim=tcon.latent_dim
)

tar_model = tar_model.to(ucon.device)
optimizer = torch.optim.AdamW(tar_model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
lr_schedule = CosineAnnealingLR(optimizer, T_max=tcon.num_epochs, eta_min=1e-6)

train_losses = []
valid_losses = []

for epoch in tqdm(range(tcon.num_epochs)):
    total_train_loss = 0
    total_valid_loss = 0

    tar_model.train()

    for batch_idx, batch in enumerate(train_loader):
        xb = batch[1]
        xc = batch[2]
        y = batch[3]

        xb = torch.flatten(xb, start_dim=1, end_dim=-1).to(ucon.device)
        xc = torch.flatten(xc, start_dim=1, end_dim=-1).to(ucon.device)
        y = y.to(ucon.device)

        with torch.no_grad():
            mean_xb, logvar_xb = vae_model.encoders["xb"](xb)
            mean_xc, logvar_xc = vae_model.encoders["xc"](xc)

            zb = vae_model.reparameterize(mean_xb, logvar_xb)
            zc = vae_model.reparameterize(mean_xc, logvar_xc)
        
            zc = zc.reshape(batch_size, 1, 8, 8)

        optimizer.zero_grad()

        loss, (z_t, outputs, logdets) = compute_loss(zc, zb)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        current_lr = lr_schedule.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

    if (epoch + 1) % tcon.sample_freq == 0:
        with torch.no_grad():
            zc_samples = tar_model.reverse(fixed_noise, fixed_zb)

        tvutils.save_image(zc_samples, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
        tvutils.save_image(tar_model.unpatchify(z_t[:100]), sample_dir / f'uspace_{epoch:03d}.png', normalize=True, nrow=10)
        
        print('sampling complete')

    tar_model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            xb = batch[1]
            xc = batch[2]
            y = batch[3]

            xb = torch.flatten(xb, start_dim=1, end_dim=-1).to(ucon.device)
            xc = torch.flatten(xc, start_dim=1, end_dim=-1).to(ucon.device)
            y = y.to(ucon.device)

            with torch.no_grad():
                mean_xb, logvar_xb = vae_model.encoders["xb"](xb)
                mean_xc, logvar_xc = vae_model.encoders["xc"](xc)
                zb = vae_model.reparameterize(mean_xb, logvar_xb)
                zc = vae_model.reparameterize(mean_xc, logvar_xc)
            
            zc = zc.reshape(64, 1, 8, 8)

            loss, (z_t, outputs, logdets) = compute_loss(zc, zb)
            total_valid_loss += loss.item()

    valid_losses.append( total_valid_loss / ((batch_idx + 1) * batch_size) )

    tqdm.write(f"\tepoch {epoch + 1} complete")
    tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
    tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

print("training complete")

torch.save(tar_model.state_dict(), ckpt_file)

# %%
# plot losses

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10,5))

ax.plot(train_losses, label="Training loss")
ax.plot(valid_losses, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_xlim(0, min(50, tcon.num_epochs))
ax.set_title(f"Total loss")
ax.legend()