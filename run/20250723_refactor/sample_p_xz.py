import torch
from torchvision import utils as tvutils

from config import UniversalConfig as ucon, ExperimentConfig as econ, VAEConfig as vcon, TARFlowConfig as tcon
from zspace.dataset import get_valid_dataset
from zspace.vae_model import get_vae
from zspace.tar_model import get_tar_model

import pathlib

if __name__ == "__main__":
    valid_dataset = get_valid_dataset(ucon)

    vae_model = get_vae(ucon, vcon, load_weights=True)
    encoders = vae_model.encoders
    decoders = vae_model.decoders

    tar_model = get_tar_model(ucon, tcon, load_weights=True)

    zbs_0 = []
    zbs_1 = []
    zbs = []
    for sample in valid_dataset:
        if len(zbs_0) == 5 and len(zbs_1) == 5:
            break
        
        _, xb, _, y = sample
        zb, _ = encoders["xb"](xb)

        if y == 0 and len(zbs_0) < 5:
            zbs_0.append(zb)
        elif y == 1 and len(zbs_1) < 5:
            zbs_1.append(zb)

    fixed_noise = torch.randn(tcon.num_classes * 10, (tcon.img_size // tcon.patch_size)**2, tcon.channel_size * tcon.patch_size ** 2, device=ucon.device)
    fixed_y = torch.arange(tcon.num_classes, device=ucon.device).view(-1, 1).repeat(1, 10).flatten()

    for zb in zbs_0:
        xc_samples_0 = tar_model.reverse(fixed_noise, zb)
    for zb in zbs_1:
        xc_samples_1 = tar_model.reverse(fixed_noise, zb)

    sample_dir = pathlib.Path(econ.output_path) / 'xc_from_zb'
    sample_dir.mkdir(exist_ok=True)

    tvutils.save_image(xc_samples_0, sample_dir / 'y=0.png', normalize=True, nrow=10)
    tvutils.save_image(xc_samples_1, sample_dir / 'y=1.png', normalize=True, nrow=10)