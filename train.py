import numpy as np
import torch
import torch.nn.functional as F
from time import time
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
from model import LoFi
from utils import *
from datasets import *
from results import evaluator
import config

torch.manual_seed(0)
np.random.seed(0)

exp_path = f'experiments/{config.task}/{config.data}_{str(config.image_size)}_{config.exp_desc}'
os.makedirs(exp_path, exist_ok= True)

# Print the experiment setup:
print(f'--> Task: {config.task}')
print(f'--> Data: {config.data}')
print(f'--> Image size: {config.image_size}')
print(f'--> Network type: {config.network}')
print(f'--> Epochs: {config.epochs}')
print(f'--> Object batch size: {config.object_batch_size}')
print(f'--> Pixel batch size train: {config.pixel_batch_size_train}')
print(f'--> Learning rate: {config.lr}')
print(f'--> Experiment directory: {exp_path}')


# Dataset:
if config.task == 'denoising':
            
    if config.data == 'radio':

        train_dataset = denoising_radio_loader(path = config.train_path,
                                        noise_level= config.noise_level,
                                        image_size = config.image_size, c = config.c_in)
        test_dataset = denoising_radio_loader(path = config.test_path,
                                        noise_level= config.noise_level,
                                        image_size = config.image_size, c = config.c_in)

    elif config.data == 'celeba-hq' or config.data == 'ten-pic':
        train_dataset = celeba_loader(dataset = 'train', path = config.train_path,
                                        noise_level= config.noise_level,
                                        image_size = config.image_size, c = config.c_in,
                                        task = config.task)
        
        test_dataset = celeba_loader(dataset = 'test', path = config.test_path,
                                        noise_level= config.noise_level,
                                        image_size = config.image_size, c = config.c_in,
                                        task = config.task)
        
        if config.ood_analysis:
            ood_dataset = celeba_loader(dataset = 'ood', path = config.ood_path,
                                        noise_level= config.noise_level,
                                        image_size = config.image_size, c = config.c_in,
                                        task = config.task)

elif config.task == 'mass':
    train_dataset = mass_loader(path = config.train_path, unet = False,
                                image_size = config.image_size,
                                y_input = config.input)
    test_dataset = mass_loader(path = config.test_path, unet = False,
                               image_size = config.image_size,
                               y_input = config.input)

elif config.task == 'LDCT':
    train_dataset = LDCT_dataloader(config.train_path, unet = False, image_size= config.image_size)
    test_dataset = LDCT_dataloader(config.test_path, unet = False, image_size= config.image_size)

    if config.ood_analysis:
        ood_dataset = LDCT_dataloader(config.ood_path, unet = False, image_size= config.image_size)


elif config.task == 'transpose':

    train_dataset = celeba_loader(dataset = 'train', path = config.train_path,
                                    noise_level= config.noise_level,
                                    image_size = config.image_size, c = config.c_in,
                                    task = config.task)
    
    test_dataset = celeba_loader(dataset = 'test', path = config.test_path,
                                    noise_level= config.noise_level,
                                    image_size = config.image_size, c = config.c_in,
                                    task = config.task)
    
    if config.ood_analysis:
        ood_dataset = celeba_loader(dataset = 'ood', path = config.ood_path,
                                    noise_level= config.noise_level,
                                    image_size = config.image_size, c = config.c_in,
                                    task = config.task)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.object_batch_size, num_workers=8, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

ntrain = len(train_loader.dataset)
n_test = len(test_loader.dataset)

n_ood = 0
if config.ood_analysis:
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=64)
    n_ood= len(ood_loader.dataset)
print('--> Number of training, test and ood samples: {}, {}, {}'.format(ntrain,n_test, n_ood))

# Loading model
model = LoFi(image_size = config.image_size, c_in = config.c_in,
               c_out = config.c_out, network = config.network,
               hidden_dim = config.hidden_dim, num_layers = config.num_layers,
               patch_shape = config.patch_shape, fourier_filtering = config.fourier_filtering,
               recep_scale = config.recep_scale,
               residual_learning = config.residual_learning,learned_geo= config.learned_geo,
               M = config.M, N = config.N, CCPG = config.CCPG,
               num_filters = config.num_filters, n_deform= config.n_deform,
               coord_deform= config.coord_deform).to(config.device)

num_param = count_parameters(model)
print('--> Number of trainable parameters of LoFi: {}'.format(num_param))
with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
    file.write('---> Number of trainable parameters of LoFi: {}'.format(num_param))
    file.write('\n')

# myloss = F.mse_loss
myloss = F.l1_loss
optimizer = Adam(model.parameters(), lr=config.lr)

checkpoint_exp_path = os.path.join(exp_path, 'LoFi.pt')
if os.path.exists(checkpoint_exp_path) and config.restore:
    checkpoint = torch.load(checkpoint_exp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('LoFi is restored...')


if config.train:
    print('Training...')

    if config.plot_per_num_epoch == -1:
        config.plot_per_num_epoch = config.epochs + 1 # only plot in the last epoch
    
    loss_plot = np.zeros([config.epochs])
    for ep in range(config.epochs):
        torch.cuda.reset_peak_memory_stats(device = config.device)
        model.train()
        t1 = time()
        
        loss_epoch = 0

        for clean_image, corrupted_image in train_loader:

            batch_size = clean_image.shape[0]
            clean_image = clean_image.to(config.device)
            corrupted_image = corrupted_image.to(config.device)

            for i in range(config.num_iterations_per_image_batch):

                coords = get_mgrid(config.image_size).reshape(-1, 2)
                coords = torch.unsqueeze(coords, dim = 0)
                coords = coords.expand(batch_size , -1, -1).to(config.device)
                
                optimizer.zero_grad()

                pixels = np.random.randint(low = 0, high = (config.image_size)**2, size = config.pixel_batch_size_train)
                batch_coords = coords[:,pixels]
                batch_image = clean_image[:,pixels]

                reconstruced_image = model(batch_coords, corrupted_image)

                mse_loss = myloss(batch_image.reshape(batch_size, -1) , reconstruced_image.reshape(batch_size, -1) )
                total_loss = mse_loss 
                total_loss.backward()
                optimizer.step()
                loss_epoch += total_loss.item()


        max_memory_used = torch.cuda.max_memory_allocated(device = config.device)
        print(f"Maximum GPU memory used in training: {max_memory_used / (1024 ** 2):.2f} MB")

        with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
            file.write(f"Maximum GPU memory used in training: {max_memory_used / (1024 ** 2):.2f} MB")
            file.write('\n')
        
        t2 = time()
        loss_epoch/= ntrain
        loss_plot[ep] = loss_epoch
        
        plt.figure()
        plt.plot(np.arange(config.epochs)[:ep] , loss_plot[:ep], 'o-', linewidth=2)
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('MSE loss')
        plt.savefig(os.path.join(exp_path, 'loss.jpg'))
        np.save(os.path.join(exp_path, 'loss.npy'), loss_plot[:ep])
        plt.close()
        
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, checkpoint_exp_path)

        print('ep: {}/{} | time: {:.0f} | Loss {:.6f} '.format(ep, config.epochs, t2-t1,loss_epoch))
        with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
            file.write('ep: {}/{} | time: {:.0f} | Loss {:.6f} '.format(ep, config.epochs, t2-t1,loss_epoch))
            file.write('\n')
        
        if ep % config.plot_per_num_epoch == 0 or (ep + 1) == config.epochs:

            evaluator(ep = ep, subset = 'test', data_loader = test_loader, model = model, exp_path = exp_path)
            if config.ood_analysis:
                evaluator(ep = ep, subset = 'ood', data_loader = ood_loader, model = model, exp_path = exp_path)




print('Evaluating...')
evaluator(ep = -1, subset = 'test', data_loader = test_loader, model = model, exp_path = exp_path)
if config.ood_analysis:
    evaluator(ep = -1, subset = 'ood', data_loader = ood_loader, model = model, exp_path = exp_path)


    
