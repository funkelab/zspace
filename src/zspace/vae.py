# %%
import os
import torch
from zspace.dataset import ToyModel
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
 
# %%
# model hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

batch_size = 100

input_dim = 3*32*32
hidden_dim = 400
latent_dim = 200

lr = 1e-3

epochs = 30

# %%
# load data

kwargs = {'num_workers': 1, 'pin_memory': True}

train_dataset = ToyModel()
test_dataset = ToyModel()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# %%
# define VAE

class Encoder(nn.Module):
    def __init__(self, input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim):
        super(Encoder, self).__init__()

        self.fc_input1 = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h = self.LeakyReLU(self.fc_input1(x))
        h = self.LeakyReLU(self.fc_input2(h))
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)

        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim):
        super(Decoder, self).__init__()
        self.fc_hidden1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.fc_hidden1(x))
        h = self.LeakyReLU(self.fc_hidden2(h))

        xa_hat = self.fc_output(h)

        return xa_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + (var * epsilon)
        return z
    
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var

# %%
# create model

encoder_xa = Encoder()
decoder_xa = Decoder()

model_xa = Model(Encoder=encoder_xa, Decoder=decoder_xa).to(device)

# %%
# define loss function and optimizer

from torch.optim import Adam
import torch.nn.functional as F

def loss_fcn(x, x_hat, mean, log_var):
    reprod_loss = F.mse_loss(x_hat, x, reduction='sum')
    D_KL = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reprod_loss + D_KL

optimizer_xa = Adam(model_xa.parameters(), lr=lr)

# %%
# train VAE

print("Start training VAE...")
model_xa.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _, _) in enumerate(train_loader):
        xa = x.view(batch_size, -1)
        xa = xa.to(device)

        optimizer_xa.zero_grad()

        xa_hat, mean, log_var = model_xa(xa)
        loss = loss_fcn(xa, xa_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage loss: ", overall_loss / ((batch_idx + 1) * batch_size))

print("Finished!")

# %%
