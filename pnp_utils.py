import torch
import numpy as np
from utils import get_mgrid
import torch.nn as nn
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as pth_transforms
import matplotlib.pyplot as plt
from utils import batch_sampling, PSNR
from tqdm import tqdm

def conv2d_from_kernel(kernel, channels, stride=1):
    """
    Returns nn.Conv2d and nn.ConvTranspose2d modules from 2D kernel, such that 
    nn.ConvTranspose2d is the adjoint operator of nn.Conv2d
    Arg:
        kernel: 2D kernel
        channels: number of image channels
    """
    kernel_size = kernel.shape
    kernel = kernel/kernel.sum()
    kernel = kernel.repeat(channels, 1, 1, 1)
    filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        # padding=kernel_size//2
    )
    # filter = nn.Conv2d(
    #     in_channels=channels, out_channels=channels,
    #     kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
    #     padding= (kernel_size[0]//2, kernel_size[1]//2)
    # )
    filter.weight.data = kernel
    filter.weight.requires_grad = False

    filter_adjoint = nn.ConvTranspose2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        # padding=kernel_size//2,
    )
    filter_adjoint.weight.data = kernel
    filter_adjoint.weight.requires_grad = False

    return filter, filter_adjoint



def pnp_admm(
        measurements, forward, forward_adjoint, denoiser, 
        step_size=1e-4, num_iter=50, max_cgiter=100, cg_tol=1e-7
    , mask = None, gt = None):
    """
    ADMM plug and play
    """
    if mask is None:
        x_h =  forward_adjoint(measurements)
    else:
        x_h =  forward_adjoint(measurements, mask)

    def conjugate_gradient(A, b, x0, max_iter, tol):
        """
        Conjugate gradient method for solving Ax=b
        """
        x = x0
        r = b-A(x)
        d = r
        for _ in range(max_iter):
            z = A(d)
            rr = torch.sum(r**2)
            alpha = rr/torch.sum(d*z)
            x += alpha*d
            r -= alpha*z
            if torch.norm(r)/torch.norm(b) < tol:
                break
            beta = torch.sum(r**2)/rr
            d = r + beta*d        
        return x

    def cg_leftside(x):
        """
        Return left side of Ax=b, i.e., Ax
        """
        if mask is None:
            out = forward_adjoint(forward(x)) + step_size*x
        else:
            out = forward_adjoint(forward(x, mask), mask) + step_size*x
        return out

    def cg_rightside(x):
        """
        Returns right side of Ax=b, i.e. b
        """
        return x_h + step_size*x

    # Start
    _ , c, h,w = x_h.shape
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    with tqdm(total=num_iter) as pbar:
        for _ in range(num_iter):
            b = cg_rightside(v-u)
            x = conjugate_gradient(cg_leftside, b, x, max_cgiter, cg_tol)

            coords = get_mgrid(h).reshape(-1, 2)
            coords = torch.unsqueeze(coords, dim = 0)
            coords = coords.expand(1 , -1, -1).to(x.device)
            # v = denoiser(coords, x+u)
            v = batch_sampling((x+u), coords, c, denoiser, s = 2**14)
            v = torch.Tensor(v).to(x.device)
            v = v.reshape(1,h,w,c).permute(0,3,1,2)

            u += (x - v)

            psnr = PSNR(gt.cpu().detach().numpy(), v.cpu().detach().numpy())
            pbar.set_description(f'PSNR: {psnr:.2f}')
            pbar.update(1)
    return v



