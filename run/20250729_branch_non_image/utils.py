import config as c

import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import umap

def get_ckpt_file(model_name, epoch=None):
    dir = pathlib.Path(c.output_dir) / f"{model_name}"
    dir.mkdir(exist_ok=True, parents=True)

    if epoch:
        return dir / f"{model_name}_epoch_{epoch}.pth"
    else:
        return dir / f"{model_name}.pth"

def get_save_dir(model_name):
    dir = pathlib.Path(c.output_dir) / f"{model_name}"
    dir.mkdir(exist_ok=True, parents=True)

    return dir

def plot_losses(train_losses, valid_losses, save_dir):
    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "losses.png")
    plt.close()

def vae_umap(vae_model, valid_loader, save_dir):
    all_zs = []
    all_labels = []

    mods = ["xa", "xb", "xc"]
    vae_model.eval()

    with torch.no_grad():
        for batch in valid_loader:
            xa, xb, xc, ys = batch
            ys = ys.tolist()

            xs = {
                "xa": xa.to(c.device).flatten(start_dim=1),
                "xb": xb.to(c.device).flatten(start_dim=1),
                "xc": xc.to(c.device).flatten(start_dim=1)
            }

            for mod in mods:
                x = xs[mod]
                means, _ = vae_model.encoders[mod](x)
                all_zs.append(means.numpy())

                for y in ys:
                    all_labels.extend([f"{mod} (y = {y})"])

    z_array = np.vstack(all_zs)

    umap_model = umap.UMAP(n_components=2)
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

    plt.savefig(save_dir / "umap.png")

def sample_p_x(tarflow_model, sample_dir, epoch=None):
    samples_per_class = 4

    zs = torch.randn(samples_per_class, c.input_dims["xc"], c.token_size)
    zs = zs.to(c.device)  

    for y in [0, 1]:
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(zs)   # xc_samples shape : (samples_per_class, input_dims["xc"], token_size)
    
        xc_samples = xc_samples.squeeze(-1)
        xc_samples = xc_samples.view(xc_samples.size(0), 8, 8)
        xc_samples = xc_samples.detach().cpu()
        
        for j, sample in enumerate(xc_samples):
            plt.figure()
            for series in sample:
                plt.plot(series.numpy())

            plt.title(f"hidden_var={y}, sample #{j}")

            if epoch is not None:
                fname = sample_dir / f"epoch_{epoch}_sample_{j}_hidden_{y}.png"
            else:
                fname = sample_dir / f"sample_{j}_hidden_{y}.png"

            plt.savefig(fname)
            plt.close()    
            
        del xc_samples

    torch.cuda.empty_cache()

def sample_p_xy(tarflow_model, sample_dir, epoch=None):
    samples_per_class = 4

    zs = torch.randn(samples_per_class, c.input_dims["xc"], c.token_size)
    zs = zs.to(c.device)  

    for y in [0, 1]:
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(zs, y)   # xc_samples shape : (samples_per_class, input_dims["xc"], token_size)
    
        xc_samples = xc_samples.squeeze(-1)
        xc_samples = xc_samples.view(xc_samples.size(0), 8, 8)
        xc_samples = xc_samples.detach().cpu()
        
        for j, sample in enumerate(xc_samples):
            plt.figure()
            for series in sample:
                plt.plot(series.numpy())

            plt.title(f"hidden_var={y}, sample #{j}")

            if epoch is not None:
                fname = sample_dir / f"epoch_{epoch}_sample_{j}_hidden_{y}.png"
            else:
                fname = sample_dir / f"sample_{j}_hidden_{y}.png"

            plt.savefig(fname)
            plt.close()    
            
        del xc_samples

    torch.cuda.empty_cache()

def sample_p_xz(dataset, vae_model, tarflow_model, sample_dir, epoch=None):
    ys_per_class = 1
    samples_per_y = 4
    y_0 = []
    y_1 = []

    for sample in dataset:
        if len(y_0) == len(y_1) == ys_per_class:
            break

        _, xb, _, hidden_var = sample
        with torch.no_grad():
            mean, logvar = vae_model.encoders["xb"](xb)
            y = vae_model.reparameterize(mean, logvar)
            y = y.to(c.device)

        if hidden_var == 0 and len(y_0) < ys_per_class:
            y_0.append(y)
        elif hidden_var == 1 and len(y_1) < ys_per_class:
            y_1.append(y)

    zs = torch.randn(samples_per_y, c.input_dims["xc"], c.token_size)
    zs = zs.to(c.device)  

    def plot_samples(xc_samples, i, hidden_var):
        xc_samples = xc_samples.squeeze(-1)
        xc_samples = xc_samples.view(xc_samples.size(0), 8, 8)
        xc_samples = xc_samples.detach().cpu()
        
        for j, sample in enumerate(xc_samples):
            plt.figure()
            for series in sample:
                plt.plot(series.numpy())

            plt.title(f"hidden_var={hidden_var}, sample #{j}, y #{i}")

            if epoch is not None:
                fname = sample_dir / f"epoch_{epoch}_sample_{j}_y_{i}_hidden_{hidden_var}.png"
            else:
                fname = sample_dir / f"sample_{j}_y_{i}_hidden_{hidden_var}.png"

            plt.savefig(fname)
            plt.close()

    for i, y in enumerate(y_0):
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(zs, y)   # shape: (samples_per_y, input_dims["xc"], token_size)

        plot_samples(xc_samples, i, hidden_var=0)

        del xc_samples
        torch.cuda.empty_cache()

    for i, y in enumerate(y_1):
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(zs, y)

        plot_samples(xc_samples, i, hidden_var=1)

        del xc_samples
        torch.cuda.empty_cache()

def sample_p_zz(dataset, vae_model, tarflow_model, sample_dir, epoch=None):
    ys_per_class = 1
    samples_per_y = 4

    y_0 = []
    y_1 = []

    for sample in dataset:
        if len(y_0) == len(y_1) == ys_per_class:
            break

        _, xb, _, hidden_var = sample
        with torch.no_grad():
            mean, logvar = vae_model.encoders["xb"](xb)
            y = vae_model.reparameterize(mean, logvar)
            y = y.to(c.device)

        if hidden_var == 0 and len(y_0) < ys_per_class:
            y_0.append(y)
        elif hidden_var == 1 and len(y_1) < ys_per_class:
            y_1.append(y)

    zs = torch.randn(samples_per_y, c.z_dim, c.token_size)
    zs = zs.to(c.device)  

    def plot_sample(xc_sample, y_idx, sample_idx, hidden_var):
        xc_sample = xc_sample.view(8, 8).detach().cpu()
        plt.figure()

        for series in xc_sample:
            plt.plot(series.numpy())

        plt.title(f"hidden_var={hidden_var}, sample #{sample_idx}, y #{y_idx}")

        if epoch is not None:
            fname = sample_dir / f"epoch_{epoch}_sample_{sample_idx}_y_{y_idx}_hidden_{hidden_var}.png"
        else:
            fname = sample_dir / f"sample_{sample_idx}_y_{y_idx}_hidden_{hidden_var}.png"

        plt.savefig(fname)
        plt.close()

    for y_idx, y in enumerate(y_0):
        with torch.no_grad():
            zc_samples = tarflow_model.reverse(zs, y)   # shape : (samples_per_y, z_dim, token_size)

        for zc_idx, zc in enumerate(zc_samples):
            zc = zc.squeeze(-1)   # shape : (z_dim)
            with torch.no_grad():
                xc_sample = vae_model.decoders["xc"](zc)   # shape : (input_dim["xc"])

            plot_sample(xc_sample, y_idx, zc_idx, hidden_var=0)

        del zc_samples
        torch.cuda.empty_cache()

    for y_idx, y in enumerate(y_1):
        with torch.no_grad():
            zc_samples = tarflow_model.reverse(zs, y)   # shape : (samples_per_y, z_dim, token_size)

        for zc_idx, zc in enumerate(zc_samples):
            zc = zc.squeeze(-1)   # shape : (z_dim)
            with torch.no_grad():
                xc_sample = vae_model.decoders["xc"](zc)   # shape : (input_dim["xc"])

            plot_sample(xc_sample, y_idx, zc_idx, hidden_var=1)

        del zc_samples
        torch.cuda.empty_cache()
