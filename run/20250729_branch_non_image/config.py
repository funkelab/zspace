import torch

# universal params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {"num_workers": 0, "pin_memory": True}
# train_seed = 42
# valid_seed = 24
batch_size = 32
z_dim = 64

output_dir = "experiment_outputs/20250731_non_image"

# VAE params
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

latent_dim = z_dim

num_epochs_vae = 100
beta_loss = 0.01
lr = 1e-3

vae_model_name = f"model_vae_{latent_dim}"

# tarflow params
token_size = 1
projection_dims = 64    # feature vector for transformer/attention block 
num_blocks = 3    
layers_per_block = 2    
nvp = True              # normalizing flow mode (non-volume preserving)
num_classes = 2
cond_dim = z_dim

noise_std = 0.05
drop_label = 0
sample_freq = 10
lr = 1e-4
weight_decay = 1e-4
num_epochs_tarflow = 300

tarflow_p_x_model_name = "model_tarflow_p_x"
tarflow_p_xy_model_name = "model_tarflow_p_xy"
tarflow_p_xz_model_name = "model_tarflow_p_xz"
tarflow_p_zz_model_name = "model_tarflow_p_zz"