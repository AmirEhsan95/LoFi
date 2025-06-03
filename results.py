import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *
import matplotlib.pyplot as plt
from mass_utils import kaiser_squires, compute_fourier_kernel, measurement
from model import LoFi
from tqdm import tqdm
from time import time



def evaluator(ep, subset, data_loader, model, exp_path):
    import config


    reconstructions_dir = os.path.join(exp_path, 'Reconstructions')
    os.makedirs(reconstructions_dir, exist_ok= True)

    print('Evaluation over {} set:'.format(subset))
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('Evaluation on {} set:'.format(subset))
        file.write('\n')

    device = model.patch_scale.device
    ngrid = int(np.sqrt(config.num_samples_visualize))

    clean_images, corrupted_images = next(iter(data_loader))
    clean_images = clean_images.to(device)
    corrupted_images = corrupted_images.to(device)
    clean_images = clean_images.reshape(-1,config.image_size, config.image_size, config.c_out)


    # Visulaizing corrupted images
    if corrupted_images.shape[1] == 2:
        corrupted_images_np = corrupted_images[:,0:1].detach().cpu().numpy()
    else:
        corrupted_images_np = corrupted_images.detach().cpu().numpy()

    corrupted_images_np = corrupted_images_np.transpose(0,2,3,1)
    corrupted_images_write = corrupted_images_np[:config.num_samples_visualize].reshape(
        ngrid, ngrid,config.image_size,config.image_size,
        config.c_out).swapaxes(1, 2).reshape(ngrid*(config.image_size),
                                         -1, config.c_out).squeeze()
    if config.c_out == 3:
        corrupted_images_write = corrupted_images_write.clip(0, 1)
    plt.imsave(os.path.join(reconstructions_dir, f'{subset}_{ep}_corrupted.png'),
        corrupted_images_write, cmap = config.cmap)

    # Visulaizing clean images:
    clean_images_np = clean_images.detach().cpu().numpy()
    clean_images_write = clean_images_np[:config.num_samples_visualize].reshape(
        ngrid, ngrid, config.image_size, config.image_size
        ,config.c_out).swapaxes(1, 2).reshape(ngrid*(config.image_size),
                                          -1, config.c_out).squeeze()
    if config.c_out == 3:
        clean_images_write = clean_images_write.clip(0, 1)
    plt.imsave(os.path.join(reconstructions_dir, f'{subset}_{ep}_clean.png'),
        clean_images_write, cmap = config.cmap)
        


    coords = get_mgrid(config.image_size).reshape(-1, 2)[None,...]
    coords = coords.expand(clean_images.shape[0] , -1, -1).to(device)

    torch.cuda.reset_peak_memory_stats(device = device)
    start = time()

    lofi_recon = batch_sampling(corrupted_images, coords,config.c_out, model, s = config.pixel_batch_size_inference)
    lofi_recon = np.reshape(lofi_recon, [-1, config.image_size, config.image_size, config.c_out])

    max_memory_used = torch.cuda.max_memory_allocated(device = device)
    print(f"Maximum GPU memory used in inference: {max_memory_used / (1024 ** 2):.2f} MB")

    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write(f"Maximum GPU memory used in inference: {max_memory_used / (1024 ** 2):.2f} MB")
        file.write('\n')

    end = time()
    print(f'inference time: {end - start}')


    lofi_recon_write = lofi_recon[:config.num_samples_visualize].reshape(
        ngrid, ngrid, config.image_size, config.image_size,
        config.c_out).swapaxes(1, 2).reshape(ngrid*(config.image_size), -1, config.c_out).squeeze()
    if config.c_out == 3:
        lofi_recon_write = lofi_recon_write.clip(0, 1)
    plt.imsave(os.path.join(reconstructions_dir, f'{subset}_{ep}_LoFi.png'),
        lofi_recon_write, cmap = config.cmap) 
     
    
    psnr_lofi = PSNR(clean_images_np, lofi_recon)
    ssim_lofi = SSIM(clean_images_np, lofi_recon)

    psnr_corrupted = PSNR(clean_images_np, corrupted_images_np)
    ssim_corrupted = SSIM(clean_images_np, corrupted_images_np)


    print('PSNR corrupted: {:.1f} | PSNR LoFi: {:.1f} | SSIM corrupted: {:.2f} | SSIM LoFi: {:.2f}'.format(
        psnr_corrupted, psnr_lofi, ssim_corrupted, ssim_lofi))
    
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('PSNR corrupted: {:.1f} | PSNR LoFi: {:.1f} | SSIM corrupted: {:.2f} | SSIM LoFi: {:.2f}'.format(
        psnr_corrupted, psnr_lofi, ssim_corrupted, ssim_lofi))
        file.write('\n')
        if subset == 'ood':
            file.write('\n')



    if config.CCPG:
        # Patch deformation analysis
        image_idx = 15
        pixel = torch.tensor([[9,92],
                                [82,73],
                                [66,54]])/128 - 0.5
        # pixel = torch.tensor([[60,110],
        #                         [33,119],
        #                         [68,77]])/128 - 0.5
        # pixel = torch.tensor([[10,16]])/128 - 0.5
        # pixel = torch.tensor([[60,110]])/128 - 0.5
        # pixel = torch.tensor([[33,119]])/128 - 0.5
        
        pixel = torch.unsqueeze(pixel, dim = 0).to(device)
        y_filtered = model.noise_suppresion_filter(y)
        patch = model(pixel, y_filtered[image_idx:image_idx+1], patch_analysis = True)

        patch = reflect_coords(patch, -1, 1)

        pixel = 2*pixel.detach().cpu().numpy().reshape(-1,2)
        patch = patch.detach().cpu().numpy().reshape(-1,2)


        pixel = ((pixel + 1) / 2) * 128
        patch = ((patch + 1) / 2) * 128
        x_pixel = [coord[1] for coord in pixel]
        y_pixel = [coord[0] for coord in pixel]

        x_patch = [coord[1] for coord in patch]
        y_patch = [coord[0] for coord in patch]

        plt.figure()
        plt.figure(figsize=(6, 6))

        plt.imshow(clean_images_np[image_idx], cmap = config.cmap)

        plt.scatter(x_patch, y_patch, color='blue', alpha = 0.2)
        plt.scatter(x_pixel, y_pixel, color='red')
        plt.title('CCPG')
        plt.axis('off')

        plt.savefig(os.path.join(reconstructions_dir,
                                 f'deformed_patch_{ep}_{subset}.png'),
                                 dpi = 1000)
        plt.close()

        

    

def evaluator_demo(data_loader, model, cmap, pixel_batch_size_inference):

    device = model.patch.device
    ngrid = 1
    num_samples_visualize = 1

    clean_images, corrupted_images = next(iter(data_loader))
    _, c_out, image_size, image_size = corrupted_images.shape
    clean_images = clean_images.to(device)
    corrupted_images = corrupted_images.to(device)
    clean_images = clean_images.reshape(-1, image_size, image_size, c_out)


    # Visulaizing corrupted images
    if corrupted_images.shape[1] == 2:
        corrupted_images_np = corrupted_images[:,0:1].detach().cpu().numpy()
    else:
        corrupted_images_np = corrupted_images.detach().cpu().numpy()

    corrupted_images_np = corrupted_images_np.transpose(0,2,3,1)
    corrupted_images_write = corrupted_images_np[:num_samples_visualize].squeeze()
    if c_out == 3:
        corrupted_images_write = corrupted_images_write.clip(0, 1)


    # Visulaizing clean images:
    clean_images_np = clean_images.detach().cpu().numpy()
    clean_images_write = clean_images_np[:num_samples_visualize].squeeze()
    if c_out == 3:
        clean_images_write = clean_images_write.clip(0, 1)
        


    coords = get_mgrid(image_size).reshape(-1, 2)[None,...]
    coords = coords.expand(clean_images.shape[0] , -1, -1).to(device)

    torch.cuda.reset_peak_memory_stats(device = device)
    start = time()

    lofi_recon = batch_sampling(corrupted_images, coords,c_out, model, s = pixel_batch_size_inference)
    lofi_recon = np.reshape(lofi_recon, [-1, image_size, image_size, c_out])

    lofi_recon_write = lofi_recon[:num_samples_visualize].squeeze()
    if c_out == 3:
        lofi_recon_write = lofi_recon_write.clip(0, 1)

    max_memory_used = torch.cuda.max_memory_allocated(device = device)
    print(f"Maximum GPU memory used in inference: {max_memory_used / (1024 ** 2):.2f} MB")

    end = time()
    print(f'inference time: {end - start}')


    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(corrupted_images_write, cmap = cmap); plt.title('Corrupted')
    plt.subplot(1,3,2); plt.imshow(lofi_recon_write, cmap = cmap); plt.title('LoFi')
    plt.subplot(1,3,3); plt.imshow(clean_images_write, cmap = cmap); plt.title('Clean')
    plt.show()
     
    
    psnr_lofi = PSNR(clean_images_np, lofi_recon)
    ssim_lofi = SSIM(clean_images_np, lofi_recon)

    psnr_corrupted = PSNR(clean_images_np, corrupted_images_np)
    ssim_corrupted = SSIM(clean_images_np, corrupted_images_np)


    print('PSNR corrupted: {:.1f} | PSNR LoFi: {:.1f} | SSIM corrupted: {:.2f} | SSIM LoFi: {:.2f}'.format(
        psnr_corrupted, psnr_lofi, ssim_corrupted, ssim_lofi))
    
    plt.close('all')