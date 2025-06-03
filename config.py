import numpy as np
import torch
device = torch.device('cuda:0')
epochs = 200 # number of epochs to train MLPatch network
object_batch_size = 64 # The number of images to be used in each training iterations
pixel_batch_size_train = 512 # Number of pixels of each image to be used in each training iteration
pixel_batch_size_inference = 64*64
num_iterations_per_image_batch = 3 # The number of training iterations over randomly selected pixels from each batch images
exp_desc = 'base' # Add a small descriptor to the experiment
image_size = 128 # Image resolution
train = True # Train or just reload to test
restore = False # Reload the exisiting 
ood_analysis = True # Evaluating the performance of model on out-of-distribution data
network = 'MultiMLP' # The network can be a 'MultiMLP' or 'MLP'
hidden_dim = 370 # Dimension of hidden layers
num_layers = 3 # Number of hidden layers
residual_learning = False # residula learning after the last layer
task = 'LDCT'  # 'denoising' or 'mass' (dark matter mapping) or 'LDCT
N = 9 # Number of chunks
M = 9 # Number of layers
patch_shape = 'round' # 'round' or 'square' 
fourier_filtering = True
num_filters = 1 # Number of Grourier filters
recep_scale = 1 # Initialization for the scale of the receptive field
lr = 1e-4 # Learning rate
num_samples_visualize = 1 # Number of reconstructed images to visulaize during inference
learned_geo = True # Learning the patch geomtery
CCPG = False # Coordinate-conditioned patch deformation (CCPG)
n_deform = 3 # Number of CCPG blocks (T in the manuscript)
coord_deform = False # Attaching INR to LoFi to transform the coordinate system
plot_per_num_epoch = 10 # Run the inference code per epochs



if task == 'LDCT':
    data = 'Chest'
    c_in = c_out = 1
    train_path = 'datasets/CT/128_180_complete_30/train'
    test_path = 'datasets/CT/128_180_complete_30/test'
    ood_path = 'datasets/CT/128_180_complete_30/outlier'
    cmap = 'gray'


elif task == 'denoising':
    noise_level = 0.15
    data = 'celeba-hq'

    if data == 'celeba-hq':

        c_in = c_out = 3
        ood_path = 'datasets/lsun-master/lsun/'
        train_path = 'datasets/celeba_hq/celeba_hq_1024_train/'
        test_path = 'datasets/celeba_hq/celeba_hq_1024_test/'
        cmap = 'gray'

    elif data == 'ten-pic': # The small training set comprising 9 training and 1 test samples

        fourier_filtering = False
        ood_analysis = False
        c_in = c_out = 3
        train_path = 'datasets/ten-pic/Train/'
        test_path = 'datasets/ten-pic/Test/'
        cmap = 'gray'

    elif data == 'radio':

        c_in = c_out = 1
        ood_analysis = False
        train_path = 'datasets/radio/train/'
        test_path = 'datasets/radio/test/'
        cmap = 'hot'

elif task == 'mass':
    data = 'kappa-TNG'
    ood_analysis = False
    input = 'ks'
    c_in = 2
    c_out = 1
    train_path = 'datasets/kappaTNG/training_data/'
    test_path = 'datasets/kappaTNG/test_data/'
    cmap = 'inferno'



elif task == 'transpose':
    data = 'celeba-hq'
    coord_deform = True
    c_in = c_out = 3
    ood_path = 'datasets/lsun-master/lsun/'
    train_path = 'datasets/celeba_hq/celeba_hq_1024_train/'
    test_path = 'datasets/celeba_hq/celeba_hq_1024_test/'
    cmap = 'gray'