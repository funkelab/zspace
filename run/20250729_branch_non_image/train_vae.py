import torch
from torch.optim import Adam
import torch.nn.functional as F

import config as c
import utils as u

from zspace.dataset import get_train_loader, get_valid_loader
from vae_model import get_vae

from tqdm import tqdm

def compute_loss(input_data, input_mod, target_data, target_mod):
    x_hat, mean, log_var = vae_model(x=input_data, input_mod=input_mod, target_mod=target_mod)

    reconst_loss = F.mse_loss(x_hat, target_data, reduction='mean')
    d_kl = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )    

    return reconst_loss, d_kl

if __name__ == "__main__":    
    train_loader = get_train_loader(c)
    valid_loader = get_valid_loader(c)

    vae_model = get_vae(c)

    optimizer = Adam(vae_model.parameters(), lr=c.lr)

    save_dir = u.get_save_dir(c.vae_model_name)
    save_dir.mkdir(exist_ok=True, parents=True)

    train_losses = []
    valid_losses = []

    self_modal_losses = []
    cross_modal_losses = []

    reconst_losses = []
    kl_losses = []

    print("Start training VAE...")

    for epoch in tqdm(range(c.num_epochs_vae)):
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
                xs[mod] = xs[mod].to(c.device)

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

            loss = reconst_loss + c.beta_loss*kl_loss

            total_self_modal_loss += self_modal_loss.item()
            total_cross_modal_loss += cross_modal_loss.item()
            total_reconst_loss += reconst_loss.item()
            total_kl_loss += kl_loss.item()
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * c.batch_size) )
        self_modal_losses.append( total_self_modal_loss / ((batch_idx + 1) * c.batch_size) )
        cross_modal_losses.append( total_cross_modal_loss / ((batch_idx + 1) * c.batch_size) )
        reconst_losses.append( total_reconst_loss / ((batch_idx + 1) * c.batch_size) )
        kl_losses.append( total_kl_loss / ((batch_idx + 1) * c.batch_size) )  

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
                    xs[mod] = xs[mod].to(c.device)

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

                loss = reconst_loss + c.beta_loss*kl_loss

                total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * c.batch_size) )

        tqdm.write(f"\tepoch {epoch + 1} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")
        
    print("training complete")
    torch.save(vae_model.state_dict(), save_dir / f"{c.vae_model_name}.pth")

    u.plot_losses(train_losses, valid_losses, save_dir)
    u.vae_umap(vae_model, valid_loader, save_dir)