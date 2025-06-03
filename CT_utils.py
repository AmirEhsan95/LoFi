import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import vgg16
from skimage.transform import radon, iradon
from scipy import optimize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import tomosipo as ts
from ts_algorithms import fbp, sirt, tv_min2d, fdk, nag_ls
import numpy as np
from datasets import *
from utils import *
import config_3D as config



def CT_sinogram(image_size = 128, n_angles = 30, 
                missing_cone = 'complete', noise_snr = 30):
    
    from skimage.transform import resize
    from skimage.transform import radon, iradon
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    import imageio
    import torch
    import torch.nn.functional as F
    

    gpu_num = 0
    device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() else 'cpu')

    train_images_dir = '../../datasets/CT/original_data/train'
    test_images_dir = '../../datasets/CT/original_data/test'
    outlier_images_dir = '../datasets/CT_brain/test_samples/images'

    train_images_names = sorted(os.listdir(train_images_dir))[:1000]
    test_images_names = sorted(os.listdir(test_images_dir))[:100]
    outlier_images_names = os.listdir(outlier_images_dir)

    n_train = len(train_images_names)
    n_test = len(test_images_names)
    n_outlier = len(outlier_images_names)
    print(n_train, n_test, n_outlier)

    data_folder = f'datasets/CT/{image_size}_{n_angles}_{missing_cone}_{noise_snr}/'
    os.makedirs(data_folder, exist_ok= True)

    train_data_folder = data_folder +  f'train/'
    os.makedirs(train_data_folder, exist_ok= True)

    test_data_folder = data_folder + f'test/'
    os.makedirs(test_data_folder, exist_ok= True)

    outlier_data_folder = data_folder + f'outlier/'
    os.makedirs(outlier_data_folder, exist_ok= True)

    np.random.seed(0)
    if missing_cone == 'horizontal':
        theta = np.linspace(30.0, 150.0, n_angles, endpoint=False)

    elif missing_cone == 'vertical':
        theta = np.linspace(-60.0, 60.0, n_angles, endpoint=False)

    else:
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    n_samples = n_test + n_train + n_outlier
    # n_samples = n_test + n_outlier

    with tqdm(total=n_samples) as pbar:
        for i in range(n_samples):

            if i < n_outlier:
                image = imageio.imread(os.path.join(outlier_images_dir, outlier_images_names[i]))
                image = (image/255.0)

            elif i < n_test + n_outlier and i >= n_outlier :
                image = np.load(os.path.join(test_images_dir, test_images_names[i-n_outlier]))
            else:
                image = np.load(os.path.join(train_images_dir, train_images_names[i-n_outlier-n_test]))

            # image = resize(image, (image_size,image_size))
            image = torch.tensor(image, dtype = torch.float32)[None,None].to(device)
            image = F.interpolate(image, size = image_size,
                                    mode = 'bilinear',
                                    antialias= True,
                                    align_corners= True)[0,0].cpu().detach().numpy()
            
            sinogram = radon(image, theta=theta, circle= False)
            noise_sigma = 10**(-noise_snr/20.0)*np.sqrt(np.mean(np.sum(
            np.square(np.reshape(sinogram, (1 , -1))) , -1)))
            noise = np.random.normal(loc = 0,
                                     scale = noise_sigma,
                                     size = np.shape(sinogram))/np.sqrt(np.prod(np.shape(sinogram)))
            sinogram += noise

            fbp = iradon(sinogram, theta=theta, circle= False)


            if i == 0:
                plt.imsave(data_folder + 'image.png', image, cmap = 'gray')
                plt.imsave(data_folder + 'sinogram.png', sinogram, cmap = 'gray')
                plt.imsave(data_folder + 'fbp.png', fbp, cmap = 'gray')
                print('First sample is saved.')
            
            if i < n_outlier:
                np.savez(outlier_data_folder + f'outlier_{i}.npz',
                         image = image,
                         sinogram = sinogram,
                         fbp = fbp)
                pbar.set_description('outlier samples...')
                pbar.update(1)

            elif i < n_test + n_outlier and i >= n_outlier :
                np.savez(test_data_folder + f'test_{i-n_outlier}.npz',
                         image = image,
                         sinogram = sinogram,
                         fbp = fbp)
                pbar.set_description('test samples...')
                pbar.update(1)

            else:
                np.savez(train_data_folder + f'train_{i-n_outlier-n_test}.npz',
                         image = image,
                         sinogram = sinogram,
                         fbp = fbp)
                pbar.set_description('train samples...')
                pbar.update(1)







def generate_projections(vol, A):
    # note angles in randians
    # n1 = vol.shape[0]
    # n2 = vol.shape[1]
    # n3 = vol.shape[2]
    # Using tomosip implementation
    projection = A(vol.permute(0,2,1))
    return projection.permute(1,0,2)

def generate_FBP(proj, A):

    rec_fbp = fbp(A, proj.permute(1,0,2)).permute(0,2,1)
    return  rec_fbp


def find_sigma_noise(SNR_value,x_ref):
    nref = torch.mean(x_ref**2)
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)


def noise_simulation(proj, noise_level):

    # Add noise
    # if noise level is a list, then sample from it
    if isinstance(noise_level, list):
        noise_level = np.random.uniform(low = min(noise_level),
                                            high = max(noise_level), size=1)[0]
    else:
        noise_level = noise_level
    # TODO : include Gaussian approximation of poisson noise
    sigma_value = find_sigma_noise(noise_level,proj)
    proj = proj + sigma_value*torch.randn_like(proj) #+  abs(proj)*torch.randn_like(proj)*alpha
    return proj

def get_wedge(size, max_angle, min_angle,radius=10):
    """
    The wedge is a 2D array of size (size,size) with a wedge of angle max_angle-min_angle
    """
    if isinstance(size, int):
        size = (size,size)

    wedge = np.zeros((size[0],size[1]))
    x = np.linspace(-1,1,size[1])
    y = np.linspace(-1,1,size[0])
    xx, yy = np.meshgrid(x, y)

    wedge[xx**2 + yy**2 < radius] = 1


    wedge[yy >np.tan(np.deg2rad(max_angle)) * xx] = 0
    wedge[yy < np.tan(np.deg2rad(min_angle)) * xx] = 0

    wedge_flip = np.fliplr(wedge)

    wedge = wedge + wedge_flip


    return wedge.T


def get_wedge_3d(size,max_angle, min_angle ,use_spherical_support = False):
    """
    Get 3D wedge with spherical support

    size: int or tuple of 3 ints
    max_angle: float (degrees)
    min_angle: float (degrees)
    rotation: float (degrees) to rotate the wedge
    use_spherical_support: bool to use spherical support or not
    Note: Default rotation is -30 degrees so to match wiht 2d when the angles are from 0 to 120
    """

    if (isinstance(size, int)):
        size = (size,size,size)
    if use_spherical_support:
        wedge_2D = get_wedge((size[-1],size[-2]), max_angle, min_angle)
    else:
        wedge_2D = get_wedge((size[-1],size[-2]), max_angle, min_angle,radius = 2)

    x = np.linspace(-1,1,size[0])
    y = np.linspace(-1,1,size[1])
    z = np.linspace(-1,1,size[2])

    xx, yy, zz = np.meshgrid(x, y, z)

    if use_spherical_support:
        ball = xx**2 + yy**2 + zz**2 < 1
        wedge_3d = wedge_2D * ball
    else:
        ball = np.ones(size) 
        wedge_3d = wedge_2D[None]*ball

    return wedge_3d 


# def main():
#     angle_max = np.pi/3
#     angle_min =  -np.pi/3
#     n_projections = 60
#     n1 = 512
#     n2 = 512
#     n3 = 256
#     noise_level = 30
#     angles = np.linspace(angle_min, angle_max, n_projections)
#     device = torch.device('cuda:' + str(config.gpu_num) if torch.cuda.is_available() else 'cpu')

#     pg = ts.parallel(angles = angles, shape =(n1, n2))
#     vg = ts.volume(shape=(n1, n3, n2))  # Reordering so that this is samle as ODL
#     A = ts.operator(vg, pg)

#     test_dataset = kidney_dataset(directory= config.data_path, subset= 'test',
#                                 n1 = n1, n2 = n2, n3 = n3,
#                                 noise_level= noise_level)
#     data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

#     vol, _ = next(iter(data_loader))
#     vol = vol.to(device)
#     _, n1, n2, n3 = vol.shape


#     proj = generate_projections(vol[0], A)
#     fbp = generate_FBP(proj, A)
#     # proj = noise_simulation(proj, noise_level)

#     print(vol.shape, vol.min(), vol.max(), vol.mean())
#     print(proj.shape, proj.min(), proj.max(), proj.mean(), torch.std(proj))


#     # GT:
#     vol_np = vol.detach().cpu().numpy()[0,:,:,n3//2]
#     plt.imsave(f'gt.png', vol_np, cmap = config.cmap)

#     k_vol = torch.fft.fftn(vol[0], dim = [0,1,2])
#     k_vol = torch.log(torch.fft.fftshift(k_vol).abs())
#     k_vol_np = k_vol.detach().cpu().numpy()[n1//2,:,:] #[:,n2//2,:] # [:,:,n3//2]
#     plt.imsave(f'k_vol.png', k_vol_np, cmap = config.cmap)

#     # FBP:
#     fbp_np = fbp.detach().cpu().numpy()[:,:,n3//2]
#     plt.imsave(f'fbp.png', fbp_np, cmap = config.cmap)


#     k_fbp = torch.fft.fftn(fbp, dim = [0,1,2])
#     k_fbp = torch.log(torch.fft.fftshift(k_fbp).abs())
#     k_fbp_np = k_fbp.detach().cpu().numpy() [n1//2,:,:] # [:,:,n3//2] # [:,n2//2,:] # [:,:,n3//2]
#     plt.imsave(f'k_fbp.png', k_fbp_np, cmap = config.cmap)


#     # Wedge applied
#     wedge  = get_wedge_3d((n1,n2,n3),max_angle=60,min_angle=-60)
#     wedge_t = torch.tensor(wedge,dtype=torch.float32, device = device)
#     vol_wedge = torch.fft.ifftn(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fftn(vol[0]))*wedge_t)).real

#     vol_wedge_np = vol_wedge.detach().cpu().numpy()[:,:,n3//2]
#     plt.imsave(f'vol_wedge.png', vol_wedge_np, cmap = config.cmap)

#     k_vol_wedge = torch.fft.fftn(vol_wedge, dim = [0,1,2])
#     k_vol_wedge = torch.log(torch.fft.fftshift(k_vol_wedge).abs() + 5.0)
#     k_vol_wedge_np = k_vol_wedge.detach().cpu().numpy()[n1//2,:,:] #[:,n2//2,:] # [:,:,n3//2]
#     plt.imsave(f'k_vol_wedge.png', k_vol_wedge_np, cmap = config.cmap)



if __name__ == '__main__':
    CT_sinogram(image_size = 128,
                missing_cone= 'complete',
                n_angles= 180,
                noise_snr= 30)
