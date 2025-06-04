# %%
import zspace
import torch
import numpy
import random

# %%
def generate_xa(y):
    """
    Generate the image xa from the binary y.
    """
    num_components = 3   # RGB
    height = 32
    width = 32
    xa = torch.zeros(num_components, height, width)
    dims = xa.shape

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if i == 0:   # R sampled from irrelevant normal distribution
                    xa[i,j,k] = torch.normal(mean=0, std=1, size=(1,)).item()
                elif i == 1:   # G sampled from relevant, low variance normal distribution
                    xa[i,j,k] = torch.normal(mean=y, std=0.1, size=(1,)).item()
                else:   # B sampled from relevant, high variance vals
                    xa[i,j,k] = torch.normal(mean=y, std=10, size=(1,)).item()

    return xa

# %%
def generate_xb(y):
    """
    Generate the bar chart probabilities xb from the binary y.
    """
    num_classes = 10
    xb = torch.zeros(num_classes)

    for i in range(xb.size(dim=0)):
        xb[i] = random.random()   # assign random val to each class

    xb = xb.softmax(dim=0)   # convert vals to probability distribution

    # sort in ascending order if y = 0, descending order if y = 1
    dec = False
    if y == 1:
        dec = True

    xb = torch.sort(xb, descending=dec)[0]

    return xb

# %%
def generate_xc(y):
    """
    Generate the time series xc from the binary c.
    """
    num_walks = 10
    num_timepoints = 50
    xc = torch.zeros(num_walks, num_timepoints)
    dims = xc.shape

    for i in range(dims[0]):
         xc[i,0] = random.random()

    bias = 0.1   # chance of moving in biased direction (dependent on y)

    for i in range(dims[0]):
        for j in range(1, dims[1]):
            if random.random() < bias:
                if y == 0:   # biased down if y = 0
                        xc[i,j] = xc[i,j-1] + torch.normal(mean=-1, std=1, size=(1,)).item()
                elif y == 1:   # biased up if y = 1
                    xc[i,j] = xc[i,j-1] + torch.normal(mean=1, std=1, size=(1,)).item()
            else:
                    xc[i,j] = xc[i,j-1] + torch.normal(mean=0, std=1, size=(1,)).item()

    # check how many walks ended in biased direction; adjust bias as needed
    # want to choose bias s.t. most walks are biased with some exceptions
    # num_biased = 0
    # for i in range(dims[0]):
    #      if y == 0 and xc[i,0] > xc[i,num_timepoints-1]:
    #             num_biased += 1
    #      elif y == 1 and xc[i,0] < xc[i,num_timepoints-1]:
    #             num_biased += 1
    # print(f'{num_biased} out of {num_walks} walks')

    return xc

# %%
print(generate_xc(1))