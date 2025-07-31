import config as c

import pathlib
import torch
import matplotlib.pyplot as plt

def get_ckpt_file(model_name, epoch=None):
    dir = pathlib.Path(c.output_dir)
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
