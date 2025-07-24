# %%
import numpy as np

# %%
# import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from notebooks.config import UniversalConfig as ucon, VAEConfig as vcon
from zspace.vae_model import JointVAE

# load model
vae_model = JointVAE(encoders=vcon.encoders, decoders=vcon.decoders, device=ucon.device).to(ucon.device)
vae_model.load_state_dict(torch.load(vcon.ckpt_file))

# %%
# define reconstruction function

def reconstruct(model: nn.Module, dataloader: DataLoader, input_mod: str):
    batch = next(iter(dataloader))
    xa = batch[0]
    xb = batch[1]
    xc = batch[2]

    for i in range(vcon.batch_size):
        xa_samp = xa[i].flatten().to(ucon.device)
        xb_samp = xb[i].flatten().to(ucon.device)
        xc_samp = xc[i].flatten().to(ucon.device)

        true_xs = {
            "xa" : xa_samp,
            "xb" : xb_samp,
            "xc" : xc_samp
        }

        with torch.no_grad():
            mean, log_var = model.encoders[input_mod](true_xs[input_mod])

            z = model.reparameterize(mean, log_var)

            xa_hat = model.decoders["xa"](z)
            xb_hat = model.decoders["xb"](z)
            xc_hat = model.decoders["xc"](z)

            reconst_xs = {
                "xa" : xa_hat,
                "xb" : xb_hat,
                "xc" : xc_hat
            }

            fig, axs = plt.subplots(1, 3, figsize=(15, 4))

            mods = ["xa", "xb", "xc"]

            for i, mod in enumerate(mods):
                x_true = true_xs[mod].cpu().detach()
                x_hat = reconst_xs[mod].cpu().detach()

                ax = axs[i]

                if mod in ["xa", "xb"]:
                    ax.bar(range(len(x_true)), x_true.numpy(), alpha=0.6, label="True")
                    ax.bar(range(len(x_hat)), x_hat.numpy(), alpha=0.6, label="Reconstructed")
                    ax.set_title(f"{mod} from {input_mod}")
                    ax.legend()

                elif mod == "xc":
                    ax.plot(x_true.numpy().T, label="True")
                    ax.plot(x_hat.numpy().T, label="Reconstructed")
                    ax.set_title(f"{mod} from {input_mod}")
                    ax.legend()

            plt.tight_layout()
            plt.show()

# %%
# sample

vae_model.eval()
mods = ["xa", "xb", "xc"]
for mod in mods:
    reconstruct(vae_model, test_loader, mod)

# %%
# plot latent space (UMAP)

import umap
import matplotlib.pyplot as plt
import numpy as np

all_zs = []
all_labels = []

mods = ["xa", "xb", "xc"]
model.eval()

with torch.no_grad():
    for batch in test_loader:
        xa, xb, xc, ys = batch
        ys = ys.tolist()

        xs = {
            "xa": xa.to(device).flatten(start_dim=1),
            "xb": xb.to(device).flatten(start_dim=1),
            "xc": xc.to(device).flatten(start_dim=1)
        }

        for mod in mods:
            x = xs[mod]
            means, _ = model.encoders[mod](x)
            all_zs.append(means.numpy())

            for y in ys:
                all_labels.extend([f"{mod} (y = {y})"])

z_array = np.vstack(all_zs)
umap_model = umap.UMAP(n_components=2, random_state=test_seed)
z_umap = umap_model.fit_transform(z_array)

unique_labels = sorted(set(all_labels))
color_palette = plt.get_cmap("Paired")
label_to_color = {}

for i, label in enumerate(unique_labels):
    color = color_palette(i)
    label_to_color[label] = color

colors = []
for label in all_labels:
    colors.append(label_to_color[label])


plt.figure(figsize=(10, 6))
plt.scatter(z_umap[:, 0], z_umap[:, 1], c=colors, alpha=0.2)
for label, color in label_to_color.items():
    plt.scatter([], [], color=color, label=label)
plt.legend()
plt.title("UMAP projection of latent space")
plt.tight_layout()
plt.show()
