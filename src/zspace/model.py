
# %%
import numpy

# %%
import zspace
import torch
from torch import nn

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
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
    def __init__(self, latent_dim, hidden_dim, output_dim):
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
    def __init__(self, Encoder, Decoder, device, seed=None):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std, generator=self.generator).to(self.device)
        eps = torch.randn(std.shape, generator=self.generator, device=self.device)
        z = mean + std * eps
        return z
    
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var
# %%
