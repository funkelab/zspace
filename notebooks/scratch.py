# %%
import zspace
import torch
import numpy

# %% [markdown] 
# this is a markdown cell

# %%
def generate_xa(y):
    """
    Generate the image xa from the binary y.
    """

    # initialize xa as an 24 x 24 img with 3 color components per pixel (RGB)
    xa = torch.zeros(3, 24, 24)
    dims = xa.shape

    # create images from different normal distributions
    irr_dist = torch.normal(mean=0, std=1, size=dims) # not dependent on y
    rel_lvar_dist = torch.normal(mean=y, std=.1, size=dims) # dependent on y with low variance
    rel_hvar_dist = torch.normal(mean=y, std=100, size=dims) # dependent on y with high variance
    
    dist_types = [irr_dist, rel_lvar_dist, rel_hvar_dist]

    # R sampled from irrelevant vals,
    # G sampled from relevant, low variance vals,
    # B sampled from relevant, high variance vals
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                xa[i,j,k] = dist_types[i][i,j,k]

    # implement without creating three reference images (unnecessary)
    # for i in range(dims[0]):
    #     for j in range(dims[1]):
    #         for k in range(dims[2]):
    #             if i == 0: 
    #                 xa[i,j,k] = torch.normal(mean=0, std=1)
    #             elif i == 1:
    #                 xa[i,j,k] = torch.normal(mean=y, std=0.1)
    #             else:
    #                 xa[i,j,k] = torch.normal(mean=y, std=10)

    return xa

# %%
print(generate_xa(1))

# %%
def generate_xb(y):
    """
    Generate the bar chart probabilities xb from the binary y.
    """
    xb = torch.zeros(10) # 10 classifications

    for i in range(xb.size(dim=0)):
        xb[i] = torch.normal(mean=0, std=1, size=(1,)).item()

    xb = xb.softmax(dim=0)

    dec = False
    if y == 1:
        dec = True

    xb = torch.sort(xb, descending=dec)[0]

    return xb

# %%
print(generate_xb(0))
print(generate_xb(1))

# %%
def generate_xc(y):
    """
    Generate the time series xc from the binary c.
    """

    return 0