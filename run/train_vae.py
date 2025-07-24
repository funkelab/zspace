# %%
import numpy as np

# %%
import torch
from torch.utils.data import DataLoader
from notebooks.config import UniversalConfig as ucon, VAEConfig as vcon
from zspace.dataset import ToyModel
from zspace.vae_model import Encoder, Decoder, JointVAE

# %%
# load data and define model

kwargs = {'num_workers': 0, 'pin_memory': True}

train_dataset = ToyModel()
valid_dataset = ToyModel()

train_gen = torch.Generator()
valid_gen = torch.Generator()

train_gen.manual_seed(ucon.train_seed)
valid_gen.manual_seed(ucon.valid_seed)

train_loader = DataLoader(dataset=train_dataset, batch_size=ucon.batch_size, generator=train_gen, shuffle=True, **kwargs)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=ucon.batch_size, generator=valid_gen, shuffle=False, **kwargs)

vae_model = JointVAE(encoders=vcon.encoders, decoders=vcon.decoders, device=ucon.device).to(ucon.device)

# %%
# define loss function and optimizer

from torch.optim import Adam
import torch.nn.functional as F

def loss_fcn(x, x_hat, mean, log_var):
    reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
    d_kl = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )

    return reconst_loss, d_kl

def compute_loss(input_data, input_mod, target_data, target_mod):
    x_hat, mean, log_var = vae_model(x=input_data, input_mod=input_mod, target_mod=target_mod)
    
    return loss_fcn(x=target_data, x_hat=x_hat, mean=mean, log_var=log_var)

def kl_anneal(epoch, total_epochs, max_beta=1.0):
    return min(max_beta, (epoch / total_epochs) * max_beta)

optimizer = Adam(vae_model.parameters(), lr=vcon.lr)

# %%
# train model

from tqdm import tqdm
import pathlib

output_path = pathlib.Path('notebook_outputs')
model_name = f'vae_{vcon.hidden_dims["xa"]}_{vcon.hidden_dims["xb"]}_{vcon.hidden_dims["xc"]}_{vcon.latent_dim}_{vcon.num_epochs}'
ckpt_file = output_path / f'model_{model_name}.pth'

train_losses = []
valid_losses = []

self_modal_losses = []
cross_modal_losses = []

reconst_losses = []
kl_losses = []

print("Start training VAE...")

for epoch in tqdm(range(vcon.num_epochs)):
    total_train_loss = 0
    total_valid_loss = 0

    total_self_modal_loss = 0
    total_cross_modal_loss = 0

    total_reconst_loss = 0 
    total_kl_loss = 0

    vae_model.train()

    for batch_idx, batch in enumerate(train_loader):
        xs = {
            "xa" : batch[0],
            "xb" : batch[1],
            "xc" : batch[2]
        }

        for mod, _ in xs.items():
            xs[mod] = torch.flatten(xs[mod], start_dim=1, end_dim=-1)
            xs[mod] = xs[mod].to(ucon.device)

        xa_sample = xs["xa"]
        xb_sample = xs["xb"]
        xc_sample = xs["xc"]

        optimizer.zero_grad()

        # losses
        reconst_xaa, kl_xa = compute_loss(xa_sample, "xa", xa_sample, "xa")
        reconst_xab, _ = compute_loss(xa_sample, "xa", xb_sample, "xb")  
        reconst_xac, _ = compute_loss(xa_sample, "xa", xc_sample, "xc") 

        reconst_xba, kl_xb = compute_loss(xb_sample, "xb", xa_sample, "xa")
        reconst_xbb, _  = compute_loss(xb_sample, "xb", xb_sample, "xb")
        reconst_xbc, _ = compute_loss(xb_sample, "xb", xc_sample, "xc")

        reconst_xca, kl_xc = compute_loss(xc_sample, "xc", xa_sample, "xa")
        reconst_xcb, _ = compute_loss(xc_sample, "xc", xb_sample, "xb")
        reconst_xcc, _ = compute_loss(xc_sample, "xc", xc_sample, "xc")

        self_modal_loss = (reconst_xaa + reconst_xbb + reconst_xcc)/3
        cross_modal_loss = (reconst_xab + reconst_xac + reconst_xba + reconst_xbc + reconst_xca + reconst_xcb)/6

        reconst_loss = self_modal_loss + cross_modal_loss
        kl_loss = kl_xa + kl_xb + kl_xc

        loss = reconst_loss + vcon.beta_loss*kl_loss

        total_self_modal_loss += self_modal_loss.item()
        total_cross_modal_loss += cross_modal_loss.item()
        total_reconst_loss += reconst_loss.item()
        total_kl_loss += kl_loss.item()
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * vcon.batch_size) )
    self_modal_losses.append( total_self_modal_loss / ((batch_idx + 1) * vcon.batch_size) )
    cross_modal_losses.append( total_cross_modal_loss / ((batch_idx + 1) * vcon.batch_size) )
    reconst_losses.append( total_reconst_loss / ((batch_idx + 1) * vcon.batch_size) )
    kl_losses.append( total_kl_loss / ((batch_idx + 1) * vcon.batch_size) )  

    vae_model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            xs = {
                "xa" : batch[0],
                "xb" : batch[1],
                "xc" : batch[2]
            }

            for mod, _ in xs.items():
                xs[mod] = torch.flatten(xs[mod], start_dim=1, end_dim=-1)
                xs[mod] = xs[mod].to(ucon.device)

            xa_sample = xs["xa"]
            xb_sample = xs["xb"]
            xc_sample = xs["xc"]        

            # losses
            reconst_xaa, kl_xa = compute_loss(xa_sample, "xa", xa_sample, "xa")
            reconst_xab, _ = compute_loss(xa_sample, "xa", xb_sample, "xb")  
            reconst_xac, _ = compute_loss(xa_sample, "xa", xc_sample, "xc") 

            reconst_xba, kl_xb = compute_loss(xb_sample, "xb", xa_sample, "xa")
            reconst_xbb, _  = compute_loss(xb_sample, "xb", xb_sample, "xb")
            reconst_xbc, _ = compute_loss(xb_sample, "xb", xc_sample, "xc")

            reconst_xca, kl_xc = compute_loss(xc_sample, "xc", xa_sample, "xa")
            reconst_xcb, _ = compute_loss(xc_sample, "xc", xb_sample, "xb")
            reconst_xcc, _ = compute_loss(xc_sample, "xc", xc_sample, "xc")

            self_modal_loss = (reconst_xaa + reconst_xbb + reconst_xcc)/3
            cross_modal_loss = (reconst_xab + reconst_xac + reconst_xba + reconst_xbc + reconst_xca + reconst_xcb)/6

            reconst_loss = self_modal_loss + cross_modal_loss
            kl_loss = kl_xa + kl_xb + kl_xc

            loss = reconst_loss + vcon.beta_loss*kl_loss

            total_valid_loss += loss.item()

    valid_losses.append( total_valid_loss / ((batch_idx + 1) * vcon.batch_size) )

    tqdm.write(f"\tepoch {epoch + 1} complete")
    tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
    tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")
    
torch.save(vae_model.state_dict(), ckpt_file)

print("finished training")

# %%
# plot losses

import matplotlib as plt

rows = 1
columns = 5

fig, axs = plt.subplots(rows, columns, figsize=(18,5))

axs[0].plot(train_losses, label="Total training loss")
axs[0].plot(valid_losses, label="Total testing loss")
axs[0].set_title(f"Total loss")
axs[0].legend()

axs[1].plot(reconst_losses, color='green', label="Reconstruction loss")
axs[1].set_title("Reconstruction training loss")

axs[2].plot(kl_losses, color='purple', label="KL divergence loss")
axs[2].set_title("KL training loss")

axs[3].plot(self_modal_losses, label="Self-modal loss")
axs[3].set_title("Self-modal training loss")

axs[4].plot(cross_modal_losses, label="Cross-modal loss")
axs[4].set_title("Cross-modal training loss")

for i in range(rows*columns):
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel("Loss")
    axs[i].set_xlim(0, min(vcon.num_epochs, 50))

plt.tight_layout()
plt.show()