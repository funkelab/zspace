import torch
from torch.optim import Adam
import torch.nn.functional as F

from config import UniversalConfig as ucon, VAEConfig as vcon, ExperimentConfig as econ

from zspace.dataset import get_train_loader, get_valid_loader
from zspace.vae_model import Encoder, Decoder, JointVAE

from tqdm import tqdm
import pathlib

def loss_fcn(x, x_hat, mean, log_var):
    reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
    d_kl = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )

    return reconst_loss, d_kl

def compute_loss(input_data, input_mod, target_data, target_mod):
    x_hat, mean, log_var = vae_model(x=input_data, input_mod=input_mod, target_mod=target_mod)
    
    return loss_fcn(x=target_data, x_hat=x_hat, mean=mean, log_var=log_var)

def kl_anneal(epoch, total_epochs, max_beta=1.0):
    return min(max_beta, (epoch / total_epochs) * max_beta)

if __name__ == "__main__":    
    train_loader = get_train_loader(ucon)
    valid_loader = get_valid_loader(ucon)

    input_dims = vcon.input_dims
    hidden_dims = vcon.hidden_dims
    latent_dim = vcon.latent_dim
    encoders = {
        "xa" : Encoder(input_dim=input_dims["xa"], hidden_dim=hidden_dims["xa"], latent_dim=latent_dim),
        "xb" : Encoder(input_dim=input_dims["xb"], hidden_dim=hidden_dims["xb"], latent_dim=latent_dim),
        "xc" : Encoder(input_dim=input_dims["xc"], hidden_dim=hidden_dims["xc"], latent_dim=latent_dim)
    }
    decoders = {
        "xa" : Decoder(mod="xa", latent_dim=latent_dim, hidden_dim=hidden_dims["xa"], output_dim=input_dims["xa"]),
        "xb" : Decoder(mod="xb", latent_dim=latent_dim, hidden_dim=hidden_dims["xb"], output_dim=input_dims["xb"]),
        "xc" : Decoder(mod="xc", latent_dim=latent_dim, hidden_dim=hidden_dims["xc"], output_dim=input_dims["xc"])
    }

    vae_model = JointVAE(encoders=encoders, decoders=decoders, device=ucon.device).to(ucon.device)

    optimizer = Adam(vae_model.parameters(), lr=vcon.lr)

    output_path = pathlib.Path(econ.output_path)
    output_path.mkdir(exist_ok=True)

    train_losses = []
    valid_losses = []

    self_modal_losses = []
    cross_modal_losses = []

    reconst_losses = []
    kl_losses = []

    print("Start training VAE...")

    for epoch in tqdm(range(vcon.num_epochs)):
        total_train_loss = 0
        total_valid_loss = 0

        total_self_modal_loss = 0
        total_cross_modal_loss = 0

        total_reconst_loss = 0 
        total_kl_loss = 0

        vae_model.train()

        for batch_idx, batch in enumerate(train_loader):
            xs = {
                "xa" : batch[0],
                "xb" : batch[1],
                "xc" : batch[2]
            }

            for mod, _ in xs.items():
                xs[mod] = torch.flatten(xs[mod], start_dim=1, end_dim=-1)
                xs[mod] = xs[mod].to(ucon.device)

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

            self_modal_loss = (reconst_xaa + reconst_xbb + reconst_xcc)/3
            cross_modal_loss = (reconst_xab + reconst_xac + reconst_xba + reconst_xbc + reconst_xca + reconst_xcb)/6

            reconst_loss = self_modal_loss + cross_modal_loss
            kl_loss = kl_xa + kl_xb + kl_xc

            loss = reconst_loss + vcon.beta_loss*kl_loss

            total_self_modal_loss += self_modal_loss.item()
            total_cross_modal_loss += cross_modal_loss.item()
            total_reconst_loss += reconst_loss.item()
            total_kl_loss += kl_loss.item()
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * ucon.batch_size) )
        self_modal_losses.append( total_self_modal_loss / ((batch_idx + 1) * ucon.batch_size) )
        cross_modal_losses.append( total_cross_modal_loss / ((batch_idx + 1) * ucon.batch_size) )
        reconst_losses.append( total_reconst_loss / ((batch_idx + 1) * ucon.batch_size) )
        kl_losses.append( total_kl_loss / ((batch_idx + 1) * ucon.batch_size) )  

        vae_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                xs = {
                    "xa" : batch[0],
                    "xb" : batch[1],
                    "xc" : batch[2]
                }

                for mod, _ in xs.items():
                    xs[mod] = torch.flatten(xs[mod], start_dim=1, end_dim=-1)
                    xs[mod] = xs[mod].to(ucon.device)

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

                self_modal_loss = (reconst_xaa + reconst_xbb + reconst_xcc)/3
                cross_modal_loss = (reconst_xab + reconst_xac + reconst_xba + reconst_xbc + reconst_xca + reconst_xcb)/6

                reconst_loss = self_modal_loss + cross_modal_loss
                kl_loss = kl_xa + kl_xb + kl_xc

                loss = reconst_loss + vcon.beta_loss*kl_loss

                total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * ucon.batch_size) )

        tqdm.write(f"\tepoch {epoch + 1} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")
        
    torch.save(vae_model.state_dict(), vcon.ckpt_file)

    print("finished training")