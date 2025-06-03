from pnp_utils import pnp_admm
import torch
from model import LoFi
import os
from datasets import denoising_radio_loader, celeba_loader
from utils import get_mgrid, PSNR, batch_sampling, SSIM
import matplotlib.pyplot as plt
import skimage
# import deepinv as dinv
from radio_utils import random_sampling, NUFFT2D_Torch, KbNuFFT2d_torch, compute_complex_sigma_noise
import numpy as np


device = torch.device('cuda:1')
image_size = 512 # Image resolution to run LoFi-ADMM
task = 'inpainting' # or 'radio'
num_iters = 90 # Number of ADMM iterations
step_sizes = [0.01,0.05,0.1] # A list of multiple step sizes to be used for LoFi-ADMM
desc = 'celeba_second'

results_folder = f'experiments/LoFi-ADMMM/{task}/{desc}/{image_size}/'
os.makedirs(results_folder, exist_ok= True)

# Print the experiment setup:
print(f'--> Task: {task}')
print(f'--> Running LoFi-ADMM in resolution: {image_size}')

###### Loading the sample
if task == 'inpainting':
    cmap = 'gray'
    p = 0.3 # The inpainting parameter
    sample = 'CelebA'
    c = 3
    if sample == 'CelebA':
        test_path = '../../datasets/celeba_hq/celeba_hq_1024_test/'
        data = celeba_loader(dataset = 'test', path = test_path,
                                noise_level= 0.15, image_size= image_size,
                                c = c)
        
        image, noisy = data[1] # Taking the second sample of the CelebA test data
        image = image.to(device)[None,...]
        noisy = noisy.to(device)[None,...]
        image = image.reshape(-1, image_size, image_size, c).permute(0,3,1,2)
        
    elif sample == 'astronaut':
        image = skimage.data.astronaut()/255.0
        image = torch.tensor(image, dtype = torch.float32).to(device)
        image = image[None,...].permute(0,3,1,2)
        noise = torch.randn(image.shape).to(device) * 0.15
        noisy = image + noise


if task == 'radio':
        cmap = 'hot'
        c = 1
        data_path = '../mlpatch/datasets/radio/test/'
        data = denoising_radio_loader(path = data_path,
                                        noise_level= 0.1,
                                        image_size = image_size, c = c)
        image, noisy = data[1] # Taking the second sample of the CelebA test data
        print(image.shape, noisy.shape)
        image = image.to(device)[None,...]
        noisy = noisy.to(device)[None,...]
        image = image.reshape(-1, image_size, image_size, c).permute(0,3,1,2)


##### Loading the model:
if task == 'inpainting':
    exp_path = 'experiments/denoising/celeba-hq_128_noise_0.15'

elif task == 'radio':

    exp_path = 'experiments/denoising/radio_128_base'

image_size_LoFi = 128 # Image resolution LoFi is trained on
network = 'MultiMLP' # The network can be a 'MultiMLP' or 'MLP'
hidden_dim = 370 # Dimension of hidden layers
num_layers = 3 # Number of hidden layers
residual_learning = False # residula learning after the last layer
N = 9 # Number of chunks
M = 9 # Number of layers
patch_shape = 'round' # 'round' or 'square' 
fourier_filtering = True
num_filters = 1
recep_scale = 1
learned_geo = True
model = LoFi(image_size = image_size_LoFi, c_in = c,
               c_out = c, network = network, hidden_dim = hidden_dim,
               num_layers = num_layers, patch_shape = patch_shape,
               fourier_filtering = fourier_filtering, recep_scale = recep_scale,
               residual_learning = residual_learning, learned_geo= learned_geo,
               M = M, N = N, CCPG = False, num_filters = num_filters,
               n_deform= 1, coord_deform= False).to(device)

checkpoint_exp_path = os.path.join(exp_path, 'LoFi.pt')
if os.path.exists(checkpoint_exp_path):
    checkpoint = torch.load(checkpoint_exp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'--> LoFi denoiser is loaded from {checkpoint_exp_path}')

##### Verify the denoising performance:
coords = get_mgrid(image_size).reshape(-1, 2)[None,...]
coords = coords.expand(1 , -1, -1).to(device)
denoised_np = batch_sampling(noisy, coords, c, model, s = 2**14)
denoised_np = denoised_np.reshape(-1, image_size,image_size,c)
image_np = image.cpu().detach().numpy().transpose(0,2,3,1)
noisy_np = noisy.cpu().detach().numpy().transpose(0,2,3,1)

if c == 3:
    noisy_np = noisy_np.clip(0,1)
    image_np = image_np.clip(0,1)
    denoised_np = denoised_np.clip(0,1)

plt.imsave(results_folder + f'Noisy.png' , noisy_np.squeeze(), cmap = cmap)
plt.imsave(results_folder + f'LoFi_denoised.png' , denoised_np.squeeze(), cmap = cmap)
plt.imsave(results_folder + f'Clean.png' , image_np.squeeze(), cmap = cmap)

psnr = PSNR(image_np, denoised_np)
ssim = SSIM(image_np, denoised_np)
print(f'LoFi performance on denoising --> PSNR: {psnr:.2f} | SSIM: {ssim:.2f}')



if task == 'inpainting':

    torch.manual_seed(1)
    mask = torch.rand(1,1,image_size,image_size).to(device)
    mask = mask < p

    def forward(x):
        return x*mask
    forward_adjoint = forward

    # Inpainting measurements
    y = forward(image)
    y_write = y.permute(0,2,3,1).squeeze().cpu().numpy().clip(0,1)
    plt.imsave(results_folder + f'Observation.png' , y_write, cmap = 'gray')

    step_sizes = [0.001, 0.005,0.01, 0.05, 0.1, 0.2, 0.5]

    for i in step_sizes:
        with torch.no_grad():
            model.eval()
            x = pnp_admm(y, forward, forward_adjoint, model,
                         num_iter=num_iters, step_size=i, gt = image)

        x_write = x.clip(0,1).permute(0,2,3,1).squeeze().cpu().numpy()
        plt.imsave(results_folder + f'LoFi-ADMM_{i}.png' , x_write, cmap = 'gray')

        psnr = PSNR(image.cpu().detach().numpy(), x.cpu().detach().numpy())
        print(f'Step size: {i} | PSNR  :{psnr:.2f}')
        with open(os.path.join(results_folder, 'results.txt'), 'a') as file:
            file.write(f'Step size: {i} | PSNR  :{psnr:.2f}')
            file.write('\n')




elif task == 'radio':

    radio_config = 'meerkat'

    if radio_config == 'meerkat':
        uv_path = "../mlpatch/plug_play/radio_utils/meerkat_simulation_1h_uv_only.npy"
        uv_data = np.load(uv_path, allow_pickle=True)[()]
        uv = np.concatenate((uv_data['uu'].reshape(-1,1), uv_data['vv'].reshape(-1,1)), axis=1)

    elif radio_config == 'random':
        uv = random_sampling(image_size**2//2)# uv coordinates between [-pi, pi]

    # set up operator
    m_op_torch = NUFFT2D_Torch(device)

    Nd = (image_size,image_size)
    Kd = (2*image_size,2*image_size) # up-sampling of 2x is typical (and also hardcoded in the operator..)
    Jd = (6,6) # 6x6 typical size for gridding kernel
    batch_size = 1
    input_snr = 30.0
    m_op_torch.plan(uv, Nd, Kd, Jd, batch_size)

    y = m_op_torch.dir_op(image) # Getting observations
    y = y.detach().cpu().numpy()

    # Define X Cai noise level
    eff_sigma = compute_complex_sigma_noise(y, input_snr)
    sigma = eff_sigma * np.sqrt(2)

    # Generate noise
    rng = np.random.default_rng(seed=0)
    n_re = rng.normal(0, eff_sigma, y[y != 0].shape)
    n_im = rng.normal(0, eff_sigma, y[y != 0].shape)
    # Add noise
    print(n_re.shape, n_im.shape)
    y[y != 0] += n_re + 1.0j * n_im
    y = torch.tensor(y, dtype = torch.complex64).to(device)
    
    # Back-projections
    BP_recon = m_op_torch.adj_op(y) 
    BP_write = BP_recon.squeeze().cpu().numpy()#.clip(0,1)
    plt.imsave(results_folder + f'BP.png' , BP_write, cmap = cmap)

    psnr_BP = PSNR(image.cpu().detach().numpy(), BP_recon.cpu().detach().numpy())
    print(f'PSNR BP: {psnr_BP: 0.2f}')

    for i in step_sizes:
        with torch.no_grad():
            model.eval()
            x = pnp_admm(y, m_op_torch.dir_op, m_op_torch.adj_op, model,
                         num_iter=num_iters, step_size=i, gt = image)
            
        x_write = x.squeeze().cpu().numpy().real
        plt.imsave(results_folder + f'LoFi-ADMM_{i}.png' , x_write, cmap = 'hot')

        psnr = PSNR(image.cpu().detach().numpy(), x.cpu().detach().numpy())
        print(f'Step size: {i} | PSNR  :{psnr:.2f}')
        with open(os.path.join(results_folder, 'results.txt'), 'a') as file:
            file.write(f'Step size: {i} | PSNR :{psnr:.2f}')
            file.write('\n')



