# %%
import numpy

# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from zspace.dataset import ToyModel
from zspace.model import Encoder, Decoder, Model
 
# %%
# model hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

while True:
    var = input("Enter xa, xb, or xc: ")
    if var in ["xa", "xb", "xc"]:
        break
    else:
        print("Invalid input; try again. ")

print("Training VAE on", var)

if var == "xa":       # xa
    input_dim = 3*1*1
    batch_comp = 0
elif var == "xb":     # xb
    input_dim = 10
    batch_comp = 1
else:                 # xc
    input_dim = 10*50
    batch_comp = 2

hidden_dim = 400
latent_dim = 200

batch_size = 64

train_seed = 42
test_seed = 24

lr = 1e-3

epochs = 30

# %%
# load data

kwargs = {'num_workers': 0, 'pin_memory': True}

train_dataset = ToyModel()
test_dataset = ToyModel()

train_gen = torch.Generator()
train_gen.manual_seed(train_seed)

test_gen = torch.Generator()
test_gen.manual_seed(test_seed)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, generator=train_gen, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, generator=test_gen, shuffle=False, **kwargs)

# %%
# create model

encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(var=var, latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)

model = Model(Encoder=encoder, Decoder=decoder, device=device).to(device)

# %%
# define loss function and optimizer

from torch.optim import Adam
import torch.nn.functional as F

def loss_fcn(x, x_hat, mean, log_var):
    if var == "xb":
        reconst_loss = F.cross_entropy(x_hat, x)
    else:
        reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
    d_kl = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )

    return reconst_loss + d_kl

optimizer = Adam(model.parameters(), lr=lr)

def reconst_loss_fcn(x, x_hat):
    reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
    return reconst_loss

def kl_loss_fcn(mean, log_var):
    d_kl = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )   
    return d_kl

# %%
# train model

from tqdm import tqdm
import matplotlib.pyplot as plt

train_losses = []
test_losses = []

reconst_losses = []
kl_losses = []

print("Start training VAE...")
model.train()

for epoch in tqdm(range(epochs)):
    total_train_loss = 0
    total_test_loss = 0

    total_reconst_loss = 0
    total_kl_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        x = batch[batch_comp] 
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.to(device)
        
        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_fcn(x, x_hat, mean, log_var)

        reconst_loss = reconst_loss_fcn(x, x_hat)
        kl_loss = kl_loss_fcn(mean, log_var)
        
        total_train_loss += loss.item()

        total_reconst_loss += reconst_loss.item()
        total_kl_loss += kl_loss.item()

        loss.backward()
        optimizer.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

    reconst_losses.append( total_reconst_loss / ((batch_idx + 1) * batch_size) )
    kl_losses.append( total_kl_loss / ((batch_idx + 1) * batch_size) )

    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pt")

    print("\tEpoch", epoch + 1, "complete!", 
          "\tAverage training loss: ", total_train_loss / ((batch_idx + 1) * batch_size))

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            x = batch[batch_comp]
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            x = x.to(device)

            x_hat, mean, log_var = model(x)
            loss = loss_fcn(x, x_hat, mean, log_var)

            total_test_loss += loss.item()

    test_losses.append( total_test_loss / ((batch_idx + 1) * batch_size) )

    print("\tAverage testing loss: ", total_test_loss / ((batch_idx + 1) * batch_size))

print("Finished!")

# %%
# plot losses

fig, axs = plt.subplots(1, 3, figsize=(18,5))

axs[0].plot(train_losses, label="Train loss")
axs[0].plot(test_losses, label="Test loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(reconst_losses, color='green', label="Reconstruction loss")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()

axs[2].plot(kl_losses, color='purple', label="KL divergence loss")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Loss")
axs[2].legend()

plt.tight_layout()
plt.show()

# %%
