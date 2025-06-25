# %%
import zspace
import numpy
import random

# %%
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# %%
class ToyModel(Dataset):
    def __init__(self, num_samples=10000, seed=0):
        self.num_samples = num_samples
        self.samples = []
        
        for i in range(num_samples):
            y = torch.randint(0, 2, (1,)).item()
            self.samples.append([self.generate_xa(y), 
                                 self.generate_xb(y), 
                                 self.generate_xc(y)])
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def generate_xa(self, y):
        """
        Generate the image xa from the binary y.
        """
        num_components = 3
        height = 1
        width = 1
        xa = torch.zeros(num_components, height, width)
        dims = xa.shape

        # color xa by sampling from different normal distributions
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    # sample R vals from irrelevant (independent from y) normal distribution
                    if i == 0:
                        xa[i,j,k] = torch.normal(mean=0, std=1, size=(1,)).item()
                    # sample G vals from relevant (dependent on y), low variance normal distribution
                    elif i == 1:
                        xa[i,j,k] = torch.normal(mean=y, std=0.1, size=(1,)).item()
                    # sample B vals from relevant, high variance normal distribution
                    else:
                        xa[i,j,k] = torch.normal(mean=y, std=10, size=(1,)).item()

        return xa      

    def generate_xb(self, y):
        """
        Generate the bar chart probabilities xb from the binary y.
        """
        num_classes = 10
        xb = torch.rand(num_classes)
        xb = xb.softmax(dim=0)

        # sort in ascending order if y = 0, descending order if y = 1
        xb = torch.sort(xb, descending=bool(y))[0]

        # add noise
        noise_level = 0.5
        noise = noise_level * torch.randn(num_classes)
        xb = xb + noise
        xb = torch.clamp(xb, min=1e-6)   # prevent non-negative vals
        xb = xb / xb.sum()   # re-normalize
        return xb

    def generate_xc(self, y):
        """
        Generate the time series xc from the binary c.
        """

        num_walks = 10
        num_timepoints = 50
        xc = torch.zeros(num_walks, num_timepoints)
        dims = xc.shape
        dims = xc.shape

        # initialize walks to random start val
        for i in range(dims[0]):
            xc[i,0] = random.random()

        # choose probability of moving in biased direction
        bias = 0.1

        # generate biased random walks with direction dependent on y
        for i in range(dims[0]):
            for j in range(1, dims[1]):
                if random.random() < bias:
                    # biased downward if y = 0
                    if y == 0:
                            xc[i,j] = xc[i,j-1] + torch.normal(mean=-1, std=1, size=(1,)).item()
                    # biased upward if y = 1
                    else:
                        xc[i,j] = xc[i,j-1] + torch.normal(mean=1, std=1, size=(1,)).item()
                else:

                        xc[i,j] = xc[i,j-1] + torch.normal(mean=0, std=1, size=(1,)).item()

        return xc

# %%
# test_model = ToyModel()
# test_loader = DataLoader(test_model, batch_size=10)
# for batch in test_loader:
#     print(batch)

# %%
