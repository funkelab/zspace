# %%
import numpy

# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from zspace.dataset import ToyModel
from zspace.model import Encoder, Decoder, JointModel

# %%
# model hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

input_dims = [3*2*2, 10, 10*50]
hidden_dim_xa = 256
hidden_dim_xb = 256
hidden_dim_xc = 1024
latent_dim = 64

batch_size = 64

train_seed = 42
test_seed = 24

beta_loss = 0.1

lr = 1e-4

num_epochs = 100

# %%
# load data

kwargs = {'num_workers': 0, 'pin_memory': True}

train_dataset = ToyModel()
test_dataset = ToyModel()

train_gen = torch.Generator()
train_gen.manual_seed(train_seed)

test_gen = torch.Generator()
test_gen.manual_seed(test_seed)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, generator=train_gen, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, generator=test_gen, shuffle=False, **kwargs)

# %%
# create joint model

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

model = JointModel(encoders=encoders, decoders=decoders, device=device).to(device)

# %%
# define loss function and optimizer

from torch.optim import Adam
import torch.nn.functional as F

optimizer = Adam(model.parameters(), lr=lr)

def loss_fcn(x, x_hat, mod, mean, log_var):
    if mod == "xb":
        reconst_loss = F.cross_entropy(x_hat, x)
    else:
       reconst_loss = F.mse_loss(x_hat, x, reduction='mean')

    d_kl = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )

    return reconst_loss, d_kl

def compute_loss(input_data, input_mod, target_data, target_mod):
    x_hat, mean, log_var = model(x=input_data, input_mod=input_mod, target_mod=target_mod)
    
    return loss_fcn(x=target_data, x_hat=x_hat, mod=target_mod, mean=mean, log_var=log_var)

def kl_anneal(epoch, total_epochs, max_beta=1.0):
    return min(max_beta, (epoch / total_epochs) * max_beta)

# %%
# train model

from tqdm import tqdm
import matplotlib.pyplot as plt

train_losses = []
test_losses = []

self_modal_losses = []
cross_modal_losses = []

reconst_losses = []
kl_losses = []

print("Start training VAE...")
model.train()

for epoch in tqdm(range(num_epochs)):
    total_train_loss = 0
    total_test_loss = 0

    total_self_modal_loss = 0
    total_cross_modal_loss = 0

    total_reconst_loss = 0
    total_kl_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        xs = {
            "xa" : batch[0],
            "xb" : batch[1],
            "xc" : batch[2]
        }

        for mod, _ in xs.items():
            xs[mod] = torch.flatten(xs[mod], start_dim=1, end_dim=-1)
            xs[mod] = xs[mod].to(device)

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

        self_modal_loss = reconst_xaa + reconst_xbb + reconst_xcc
        cross_modal_loss = reconst_xab + reconst_xac + reconst_xba + reconst_xbc + reconst_xca + reconst_xcb

        reconst_loss = self_modal_loss + cross_modal_loss
        kl_loss = kl_xa + kl_xb + kl_xc

        loss = reconst_loss + beta_loss*kl_loss

        total_self_modal_loss += self_modal_loss.item()
        total_cross_modal_loss += cross_modal_loss.item()
        total_reconst_loss += reconst_loss.item()
        total_kl_loss += kl_loss.item()
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )
    self_modal_losses.append( total_self_modal_loss / ((batch_idx + 1) * batch_size) )
    cross_modal_losses.append( total_cross_modal_loss / ((batch_idx + 1) * batch_size) )
    reconst_losses.append( total_reconst_loss / ((batch_idx + 1) * batch_size) )
    kl_losses.append( total_kl_loss / ((batch_idx + 1) * batch_size) )  

    print("\tEpoch", epoch + 1, "complete!", 
          "\tAverage training loss: ", total_train_loss / ((batch_idx + 1) * batch_size))

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            xs = {
                "xa" : batch[0],
                "xb" : batch[1],
                "xc" : batch[2]
            }

            for mod, _ in xs.items():
                xs[mod] = torch.flatten(xs[mod], start_dim=1, end_dim=-1)
                xs[mod] = xs[mod].to(device)

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

            reconst_loss = reconst_xaa + reconst_xab + reconst_xac + reconst_xba + reconst_xbb + reconst_xbc + reconst_xca + reconst_xcb + reconst_xcc
            kl_loss = kl_xa + kl_xb + kl_xc
            loss = reconst_loss + beta_loss*kl_loss

            total_test_loss += loss.item()

    test_losses.append( total_test_loss / ((batch_idx + 1) * batch_size) )

    print("\tAverage testing loss: ", total_test_loss / ((batch_idx + 1) * batch_size))

print("Finished!")


# %%
# plot losses

rows = 1
columns = 5

fig, axs = plt.subplots(rows, columns, figsize=(18,5))

axs[0].plot(train_losses, label="Total training loss")
axs[0].plot(test_losses, label="Total testing loss")
axs[0].set_title(f"Total loss (beta = {beta_loss})")
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
    axs[i].set_xlim(0, min(num_epochs, 250))

plt.tight_layout()
plt.show()

# %%
# define reconstruction graph functions
import matplotlib.pyplot as plt
import torch

def reconstruct(true_xs: dict[str, torch.Tensor], reconst_xs: dict[str, torch.Tensor], input: str):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    mods = ["xa", "xb", "xc"]

    for i, mod in enumerate(mods):
        x_true = true_xs[mod].cpu().detach()
        x_hat = reconst_xs[mod].cpu().detach()

        ax = axs[i]

        if mod == "xa" or mod == "xb":
            ax.bar(range(len(x_true)), x_true.numpy(), alpha=0.6, label="True")
            ax.bar(range(len(x_hat)), x_hat.numpy(), alpha=0.6, label="Reconstructed")
            ax.set_title(f"{mod} from {input}")
            ax.legend()

        elif mod == "xc":
            ax.plot(x_true.numpy().T, label="True")
            ax.plot(x_hat.numpy().T, label="Reconstructed")
            ax.set_title(f"{mod} from {input}")
            ax.legend()

    plt.tight_layout()
    plt.show()

# %%
# sample xas, xbs, xcs, and decode in all three modalities

model.eval()

batch = next(iter(test_loader))
xa = batch[0]
xb = batch[1]
xc = batch[2]

for i in range(batch_size):
    xa_samp = xa[i].flatten().to(device)
    xb_samp = xb[i].flatten().to(device)
    xc_samp = xc[i].flatten().to(device)

    true_xs = {
        "xa" : xa_samp,
        "xb" : xb_samp,
        "xc" : xc_samp
    }

    with torch.no_grad():
        # xa input
        mean, log_var = model.encoders["xa"](xa_samp)
        z = model.reparameterize(mean, log_var)

        xa_hat = model.decoders["xa"](z)
        xb_hat = model.decoders["xb"](z)
        xc_hat = model.decoders["xc"](z)

        reconst_xs = {
            "xa" : xa_hat,
            "xb" : xb_hat,
            "xc" : xc_hat
        }

        reconstruct(true_xs, reconst_xs, "xa")


# %%
# plot latent space (umap)
