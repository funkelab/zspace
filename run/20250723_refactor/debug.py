import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils as tvutils

from tqdm import tqdm
import pathlib

from zspace.dataset import get_valid_dataset, get_train_loader, get_valid_loader
from zspace.tarflow_model import get_tarflow_model
from zspace.vae_model import get_vae
from config import UniversalConfig as ucon, TARFlowConfig as tcon, VAEConfig as vcon
from train_tarflow_p_xz import compute_loss

if __name__ == "__main__":

    ckpt_file = pathlib.Path(tcon.save_dir_p_xz) / "checkpoints/tarflow_model_debug_epoch_20.pth"
    tarflow_model = get_tarflow_model(ucon, tcon, ckpt_file=ckpt_file)

    save_dir = pathlib.Path(tcon.save_dir_p_xz)

    fixed_noise = torch.randn(
        1, 
        2**2, 
        1, 
        device=ucon.device)
    
    z_0 = torch.zeros((1, 1), dtype=torch.float32)
    z_1 = torch.ones((1, 1), dtype=torch.float32)
    
    x_0 = tarflow_model.reverse(fixed_noise, z_0)
    x_1 = tarflow_model.reverse(fixed_noise, z_1)

    print(f"{x_0=}, {x_1=}")