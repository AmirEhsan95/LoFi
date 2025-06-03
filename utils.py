import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import vgg16
from skimage.transform import radon, iradon
from scipy import optimize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from time import time


def reflect_coords(ix, min_val, max_val):

    pos_delta = ix[ix>max_val] - max_val

    neg_delta = min_val - ix[ix < min_val]

    ix[ix>max_val] = ix[ix>max_val] - 2*pos_delta
    ix[ix<min_val] = ix[ix<min_val] + 2*neg_delta

    return ix



def SSIM(x_true , x_pred):
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += ssim(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min(),
                  channel_axis = 2)
        
    return s/np.shape(x_pred)[0]


def PSNR(x_true , x_pred):
    
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def batch_sampling(image_recon, coords, c, model, s = 512):
    outs = np.zeros([np.shape(coords)[0], np.shape(coords)[1], c])
    with torch.no_grad():
        for i in range(np.shape(coords)[1]//s):
            batch_coords = coords[:,i*s: (i+1)*s]
            out = model(batch_coords, image_recon).detach().cpu().numpy()
            outs[:,i*s: (i+1)*s] = out
        
    return outs


    
def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1)[None,...].astype(np.float32)
    pixel_coords /= (sidelen-1)   
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords

