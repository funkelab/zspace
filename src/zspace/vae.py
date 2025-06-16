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

batch_size = 64

input_dim = 3*32*32
hidden_dim = 400
latent_dim = 200

lr = 1e-3

epochs = 30

# %%
# load data

kwargs = {'num_workers': 0, 'pin_memory': True}

train_dataset = ToyModel()
test_dataset = ToyModel()

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# %%
# create model

encoder_xa = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder_xa = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)

model_xa = Model(Encoder=encoder_xa, Decoder=decoder_xa, device=device).to(device)

# %%
# define loss function and optimizer

from torch.optim import Adam
import torch.nn.functional as F

def loss_fcn(x, x_hat, mean, log_var):
    reprod_loss = F.mse_loss(x_hat, x, reduction='sum')
    D_KL = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )

    return reprod_loss + D_KL

optimizer_xa = Adam(model_xa.parameters(), lr=lr)

# %%
# train VAE

from tqdm import tqdm
import matplotlib.pyplot as plt

train_losses = []
test_losses = []

print("Start training VAE...")
model_xa.train()

for epoch in tqdm(range(epochs)):
    total_train_loss = 0
    total_test_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        xa = batch[0]   # batch[0] shape: [batch_size, 3, 32, 32]
        xa = torch.flatten(xa, start_dim=1, end_dim=-1)
        xa = xa.to(device)

        optimizer_xa.zero_grad()

        xa_hat, mean, log_var = model_xa(xa)
        loss = loss_fcn(xa, xa_hat, mean, log_var)

        total_train_loss += loss.item()

        loss.backward()
        optimizer_xa.step()

    train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

    print("\tEpoch", epoch + 1, "complete!", 
          "\tAverage training loss: ", total_train_loss / ((batch_idx + 1) * batch_size))

    model_xa.eval()
    with torch.no_grad():
        for batch in test_loader:
            xa = batch[0]
            xa = torch.flatten(xa, start_dim=1, end_dim=-1)
            xa = xa.to(device)

            xa_hat, mean, log_var = model_xa(xa)
            loss = loss_fcn(xa, xa_hat, mean, log_var)

            total_test_loss += loss.item()

    test_losses.append( total_test_loss / ((batch_idx + 1) * batch_size) )

    print("\tAverage test loss: ", total_test_loss / ((batch_idx + 1) * batch_size))

print("Finished!")

# %%
# plot losses
plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
