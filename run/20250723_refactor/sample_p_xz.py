import torch
from torchvision import utils as tvutils

from config import UniversalConfig as ucon, ExperimentConfig as econ, VAEConfig as vcon, TARFlowConfig as tcon
from zspace.dataset import get_valid_dataset
from zspace.vae_model import get_vae
from zspace.tarflow_model import get_tarflow_model

import pathlib

if __name__ == "__main__":
    valid_dataset = get_valid_dataset(ucon)
    vae_model = get_vae(ucon, vcon, ckpt_file=vcon.ckpt_file)
    tarflow_model = get_tarflow_model(ucon, tcon, tcon.ckpt_file_p_xz)

    sample_dir = pathlib.Path(econ.output_path) / "xc_from_zb"
    sample_dir.mkdir(exist_ok=True)

    num_zbs = 2
    zbs_0 = []
    zbs_1 = []

    for sample in valid_dataset:
        print("sampling zbs")
        if len(zbs_0) == num_zbs and len(zbs_1) == num_zbs:
            break
        
        _, xb, _, y = sample
        zb, _ = vae_model.encoders["xb"](xb)

        if y == 0 and len(zbs_0) < num_zbs:
            zbs_0.append(zb)
        elif y == 1 and len(zbs_1) < num_zbs:
            zbs_1.append(zb)

    print("creating noise")
    num_samples_per_class = 10
    fixed_noise = torch.randn(
        tcon.num_classes * num_samples_per_class, 
        (tcon.img_size // tcon.patch_size)**2, 
        tcon.channel_size * tcon.patch_size ** 2, 
        device=ucon.device)
    fixed_y = torch.arange(tcon.num_classes, device=ucon.device).view(-1, 1).repeat(1, num_samples_per_class ).flatten()

    # Generate and save samples for y=0
    for i, zb in enumerate(zbs_0):
        print(f"reconstructing xcs with y=0, zb={zb}")
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(fixed_noise, zb)
        # Save immediately and clear memory
        save_path = sample_dir / f'y=0_{i}.png'
        tvutils.save_image(xc_samples, save_path, normalize=True, nrow=num_samples_per_class)
        del xc_samples
        torch.cuda.empty_cache()

    # Generate and save samples for y=1s
    for i, zb in enumerate(zbs_1):
        print(f"reconstructing xcs with y=1, zb={zb}")
        with torch.no_grad():
            xc_samples = tarflow_model.reverse(fixed_noise, zb)
        save_path = sample_dir / f'y=1_{i}.png'
        tvutils.save_image(xc_samples, save_path, normalize=True, nrow=num_samples_per_class)
        del xc_samples
        torch.cuda.empty_cache()

    # xc_samples_0 = []
    # xc_samples_1 = []

    # for zb in zbs_0:
    #     print("reconstructing xcs with y=0")
    #     print(f"{zb=}")
    #     xc_samples_0 += tarflow_model.reverse(fixed_noise, zb)

    # for zb in zbs_1:
    #     print("reconstructing xcs with y=1")
    #     print(f"{zb=}")
    #     xc_samples_1 += tarflow_model.reverse(fixed_noise, zb)

    # tvutils.save_image(xc_samples_0, sample_dir / 'y=0.png', normalize=True, nrow=10)
    # tvutils.save_image(xc_samples_1, sample_dir / 'y=1.png', normalize=True, nrow=10)