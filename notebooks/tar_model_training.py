# %%
import numpy as np

# %%
import os

import torch
from torch import nn
from torch import utils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision as tv

from tqdm import tqdm
import pathlib

from zspace.dataset import ToyModel
from zspace.tar_model import Model
from zspace.vae_model import Encoder, Decoder, JointVAE

# %%
# model hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

batch_size = 64
lr = 1e-4
train_seed = 42
valid_seed = 24
num_epochs = 100

# %%
# load data

kwargs = {'num_workers': 0, 'pin_memory': True}

train_dataset = ToyModel()
valid_dataset = ToyModel()

train_gen = torch.Generator()
valid_gen = torch.Generator()
train_gen.manual_seed(train_seed)
valid_gen.manual_seed(valid_seed)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, generator=train_gen, shuffle=True, **kwargs)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, generator=valid_gen, shuffle=False, **kwargs)


# %%
# define loss function

def compute_loss(x, y):
    z, outputs, logdets = tar_model(x, y)
    loss = tar_model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)

# %%
# train model to learn p(xc|y)

in_channels = 1         
img_size = 8           
patch_size = 1          
channels = 64          
num_blocks = 3          
layers_per_block = 2    
nvp = True              # normalizing flow mode (non-volume preserving)
num_classes = 2         # binary y
channel_size = 1

noise_std = 0.05
drop_label = 0
sample_freq = 10

fixed_noise = torch.randn(num_classes * 10, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
fixed_y = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, 10).flatten()

train_losses = []
valid_losses = []

model_name = f'tar_{patch_size}_{channels}_{num_blocks}_{layers_per_block}_{noise_std:.2f}'
output_path = pathlib.Path('notebook_outputs')
sample_dir = output_path / f'xc_samples_{model_name}'
ckpt_file = output_path / f'xc_model_{model_name}.pth'
sample_dir.mkdir(exist_ok=True, parents=True)    

tar_model = Model(
    in_channels=in_channels,
    img_size=img_size,
    patch_size=patch_size,
    channels=channels,
    num_blocks=num_blocks,
    layers_per_block=layers_per_block,
    nvp=nvp,
    num_classes=num_classes
)

tar_model = tar_model.to(device)
optimizer = torch.optim.AdamW(tar_model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

for epoch in tqdm(range(num_epochs)):
    total_train_loss = 0
    total_valid_loss = 0

    tar_model.train()

    for batch_idx, batch in enumerate(train_loader):
        xc = batch[2].unsqueeze(1).to(device)
        y = batch[3].to(device)      

        eps = noise_std * torch.randn_like(xc)
        xc = xc + eps

        # mask = (torch.rand(y.size(0), device=device) < drop_label).int()   # -1 denotes dropped class
        # y = (1 - mask) * y - mask

        optimizer.zero_grad()

        loss, (z_t, outputs, logdets) = compute_loss(xc, y)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        current_lr = lr_schedule.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

    if (epoch + 1) % sample_freq == 0:
        with torch.no_grad():
            xc_samples = tar_model.reverse(fixed_noise, fixed_y)

        tv.utils.save_image(xc_samples, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
        tv.utils.save_image(tar_model.unpatchify(z_t[:100]), sample_dir / f'uspace_{epoch:03d}.png', normalize=True, nrow=10)
        
        print('sampling complete')

    tar_model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            xc = batch[2].unsqueeze(1).to(device)
            y = batch[3].to(device)

            eps = noise_std * torch.randn_like(xc)
            xc = xc + eps

            # mask = (torch.rand(y.size(0), device=device) < drop_label).int()   # -1 denotes dropped class
            #y = (1 - mask) * y - mask

            loss, (z_t, outputs, logdets) = compute_loss(xc, y)
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

fig, ax = plt.subplots(1, 1, figsize=(18,5))

ax.plot(train_losses, label="Training loss")
ax.plot(valid_losses, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"Total loss")
ax.legend()

# %%
# evaluate model as classifier
num_correct = 0
num_examples = 0

tar_model.eval()

for batch_idx, batch in enumerate(valid_loader):
    xc = batch[2].unsqueeze(1).to(device)
    y = batch[3].to(device)

    eps = noise_std * torch.randn_like(xc)
    xc = xc.repeat(num_classes, 1, 1, 1)
    y_ = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, y.size(0)).flatten()
    
    with torch.no_grad():
        z, outputs, logdets = tar_model(xc, y_)
        losses = 0.5 * z.pow(2).mean(dim=[1, 2]) - logdets # keep the batch dimension
        pred = losses.reshape(num_classes, y.size(0)).argmin(dim=0)

    num_correct += (pred == y).sum()
    num_examples += y.size(0)

print(f'accuracy: {100 * num_correct / num_examples:.2f}%')

# %%
# train model to learn p(xc|zb)

input_dims = [3*2*2, 10, 8*8]
hidden_dim_xa = 256
hidden_dim_xb = 256
hidden_dim_xc = 1024
latent_dim = 64

encoders = {
    "xa" : Encoder(input_dim=input_dims[0], hidden_dim=hidden_dim_xa, latent_dim=latent_dim),
    "xb" : Encoder(input_dim=input_dims[1], hidden_dim=hidden_dim_xb, latent_dim=latent_dim),
    "xc" : Encoder(input_dim=input_dims[2], hidden_dim=hidden_dim_xc, latent_dim=latent_dim)
}

decoders = {
    "xa" : Decoder(mod="xa", latent_dim=latent_dim, hidden_dim=hidden_dim_xa, output_dim=input_dims[0]),
    "xb" : Decoder(mod="xb", latent_dim=latent_dim, hidden_dim=hidden_dim_xb, output_dim=input_dims[1]),
    "xc" : Decoder(mod="xc", latent_dim=latent_dim, hidden_dim=hidden_dim_xc, output_dim=input_dims[2])
}

model_name = f'vae_{hidden_dim_xa}_{hidden_dim_xb}_{hidden_dim_xc}_{latent_dim}_{num_epochs}'
vae_ckpt = output_path / f'model_{model_name}.pth'

vae_model = JointVAE(encoders=encoders, decoders=decoders, device=device).to(device)
vae_model.load_state_dict(torch.load(vae_ckpt))

in_channels = 1         
img_size = 8           
patch_size = 1          
channels = 64          
num_blocks = 3          
layers_per_block = 2    
nvp = True              # normalizing flow mode (non-volume preserving)
num_classes = 2         # binary y
channel_size = 1

noise_std = 0.05
drop_label = 0
sample_freq = 10

fixed_noise = torch.randn(num_classes * 10, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
fixed_y = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, 10).flatten()

train_losses = []
valid_losses = []

model_name = f'tar_{patch_size}_{channels}_{num_blocks}_{layers_per_block}_{noise_std:.2f}'
output_path = pathlib.Path('notebook_outputs')
sample_dir = output_path / f'xc_from_zb_samples_{model_name}'
ckpt_file = output_path / f'xc_from_zb_model_{model_name}.pth'
sample_dir.mkdir(exist_ok=True, parents=True)    

tar_model = Model(
    in_channels=in_channels,
    img_size=img_size,
    patch_size=patch_size,
    channels=channels,
    num_blocks=num_blocks,
    layers_per_block=layers_per_block,
    nvp=nvp,
    num_classes=num_classes
)

tar_model = tar_model.to(device)
optimizer = torch.optim.AdamW(tar_model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

for epoch in tqdm(range(num_epochs)):
    total_train_loss = 0
    total_valid_loss = 0

    tar_model.train()

    for batch_idx, batch in enumerate(train_loader):
        xb = batch[1]
        xc = batch[2]

        xb = torch.flatten(xb, start_dim=1, end_dim=-1).to(device)
        xc = xc.unsqueeze(1).to(device)

        with torch.no_grad():
            mean, logvar = vae_model.encoders["xb"](xb)
            zb = vae_model.reparameterize(mean, logvar)
        
        eps = noise_std * torch.randn_like(xc)
        xc = xc + eps

        # mask = (torch.rand(y.size(0), device=device) < drop_label).int()   # -1 denotes dropped class
        # y = (1 - mask) * y - mask

        optimizer.zero_grad()

        loss, (z_t, outputs, logdets) = compute_loss(xc, zb)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        current_lr = lr_schedule.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

    if (epoch + 1) % sample_freq == 0:
        with torch.no_grad():
            xc_samples = tar_model.reverse(fixed_noise, fixed_y)

        tv.utils.save_image(xc_samples, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
        tv.utils.save_image(tar_model.unpatchify(z_t[:100]), sample_dir / f'uspace_{epoch:03d}.png', normalize=True, nrow=10)
        
        print('sampling complete')

    tar_model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            xc = batch[2].unsqueeze(1).to(device)
            y = batch[3].to(device)

            eps = noise_std * torch.randn_like(xc)
            xc = xc + eps

            # mask = (torch.rand(y.size(0), device=device) < drop_label).int()   # -1 denotes dropped class
            #y = (1 - mask) * y - mask

            loss, (z_t, outputs, logdets) = compute_loss(xc, y)
            total_valid_loss += loss.item()

    valid_losses.append( total_valid_loss / ((batch_idx + 1) * batch_size) )

    tqdm.write(f"\tepoch {epoch + 1} complete")
    tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
    tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

print("training complete")

torch.save(tar_model.state_dict(), ckpt_file)
# %%
