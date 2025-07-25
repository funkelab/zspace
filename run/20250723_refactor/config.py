import torch
import pathlib

class UniversalConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"num_workers": 0, "pin_memory": True}
    train_seed = 42
    valid_seed = 24
    batch_size = 64
    z_dim = 64

class ExperimentConfig:
    output_path = "experiment_outputs"

class VAEConfig:
    input_dims = {
        "xa" : 3*2*2,
        "xb" : 10,
        "xc" : 8*8
    }
    hidden_dims = {
        "xa" : 256,
        "xb" : 256,
        "xc" : 1024
    }
    latent_dim = UniversalConfig.z_dim

    num_epochs = 100
    beta_loss = 0.01
    lr = 1e-3

    model_name = f'vae_{hidden_dims["xa"]}_{hidden_dims["xb"]}_{hidden_dims["xc"]}_{latent_dim}_{num_epochs}'
    ckpt_file = ExperimentConfig.output_path + f'/model_{model_name}.pth'

class TARFlowConfig:
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
    cond_dim = UniversalConfig.z_dim

    num_epochs = 100
    lr = 1e-4
    weight_decay = 1e-4

    model_name_p_xy = "tarflow_p_xy"
    model_name_p_xz = "tarflow_p_xz"
    save_dir_p_xy = pathlib.Path(ExperimentConfig.output_path) / f"model_{model_name_p_xy}"
    save_dir_p_xz = pathlib.Path(ExperimentConfig.output_path) / f"model_{model_name_p_xz}"