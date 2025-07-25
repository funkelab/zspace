import torch
from torchvision import utils as tvutils

from config import UniversalConfig as ucon, ExperimentConfig as econ, VAEConfig as vcon, TARFlowConfig as tcon
from zspace.dataset import get_valid_dataset
from zspace.vae_model import get_vae
from zspace.tarflow_model import get_tarflow_model

import pathlib

if __name__ == "__main__":
    valid_dataset = get_valid_dataset(ucon)
    vae_model = get_vae(ucon, vcon, load_weights=True)
    tarflow_model = get_tarflow_model(ucon, tcon, ckpt_file=tcon.ckpt_file_p_xy)

    fixed_noise = torch.randn(tcon.num_classes * 10, (tcon.img_size // tcon.patch_size)**2, tcon.channel_size * tcon.patch_size ** 2, device=ucon.device)
    fixed_y = torch.arange(tcon.num_classes, device=ucon.device).view(-1, 1).repeat(1, 10).flatten()

    sample_dir = pathlib.Path(econ.output_path) / "xc_from_zb"
    sample_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        xc_samples = tarflow_model.reverse(fixed_noise, fixed_y)
        tvutils.save_image(xc_samples, sample_dir / "xc_samples_p_xy.png", normalize=True, nrow=10)