import torch
from utils import *
import numpy as np
import os
import config
import matplotlib.pyplot as plt
from mass_utils import compute_fourier_kernel, kaiser_squires, measurement
from torchvision import transforms
import torchvision
import torch
from utils import *
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import imageio



class LDCT_dataloader(torch.utils.data.Dataset):

    def __init__(self, directory, unet = False, image_size = 128):

        self.directory = directory

        self.name_list = sorted(os.listdir(self.directory))
        self.unet = unet
        self.image_size = image_size


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        file = np.load(os.path.join(self.directory,file_name))
        image = file['image']
        fbp = file['fbp']
        image = torch.tensor(image, dtype = torch.float32)
        fbp = torch.tensor(fbp, dtype = torch.float32)

        image = F.interpolate(image[None,None,...], size = self.image_size,
                              mode = 'bilinear',
                              antialias= True,
                              align_corners= True)[0,0]
        
        fbp = F.interpolate(fbp[None,None,...], size = self.image_size,
                              mode = 'bilinear',
                              antialias= True,
                              align_corners= True)[0,0]
        
        if self.unet:

            return image[None,...], fbp[None,...]
        
        else:
            image = image.reshape(-1, 1)
            return image, fbp[None,...]





class mass_loader(torch.utils.data.Dataset):

    def __init__(self, path, unet = False, image_size = 128,
                 y_input = 'ks'):

        self.path = path
        self.name_list = os.listdir(self.path)
        self.fourier_kernel = compute_fourier_kernel(image_size)
        self.image_size = image_size
        self.unet = unet
        self.y_input = y_input
        
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        image_size = self.image_size
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        image_1024 = np.load(os.path.join(self.path,file_name))
        image_1024 = torch.tensor(image_1024, dtype = torch.float32)[None,...]

        if image_size == 1024:
            image = image_1024
        
        else:
            image = torchvision.transforms.CenterCrop(image_size)(image_1024)

        if self.y_input == 'ks':
            y = kaiser_squires(image, self.fourier_kernel)

        elif self.y_input == 'shear':
            y = measurement(image, self.fourier_kernel)
        elif self.y_input == 'noisy':
            max_val = image.abs().max()
            noise = torch.randn(image.shape) * 0.15 * max_val
            y = image + noise
        
        if not self.unet:
            image = image.permute([1,2,0]).reshape(-1, config.c_out)
        return image, y

    


class celeba_loader(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_level = 0.3, path = None,
                 unet = False, image_size = 128, c = 3, task = 'denoising',
                 translate = False, rotate = False):

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])


        if dataset == 'train':
            self.img_dataset = ImageFolder(path, self.transform)
            self.shift_seed = len(self.img_dataset)
            

        elif dataset == 'test':
            self.img_dataset = ImageFolder(path, self.transform)
            self.shift_seed = 0
            

        elif dataset == 'ood':
            lsun_class = ['bedroom_val']
            self.img_dataset = torchvision.datasets.LSUN(path,
                classes=lsun_class, transform=self.transform)
            self.shift_seed = len(self.img_dataset)
            # self.img_dataset = ImageFolder(config.ood_path, self.transform)
        
        
        self.image_shape = [c, image_size, image_size]
        self.unet = unet
        self.noise_level = noise_level
        self.c = c
        self.subset = dataset
        self.task = task
        self.translate = translate
        self.rotate = rotate

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img = self.img_dataset[item][0]
        img = transforms.ToPILImage()(img)
        img = self.transform(img)

        if self.task == 'denoising':
            torch.manual_seed(item + self.shift_seed)
            noise = torch.randn(self.image_shape) * self.noise_level
            y = img + noise

        
        elif self.task == 'transpose':
            y = img.permute(0,2,1)

        if not self.unet:
            img = img.permute([1,2,0]).reshape(-1, self.c)

        return img, y




class denoising_radio_loader(torch.utils.data.Dataset):
    def __init__(self, noise_level = 0.3, path = None,
                 unet = False, image_size = 128, c = 3):

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])

        self.path = path
        self.name_list = os.listdir(self.path)
        
        self.image_shape = [c, image_size, image_size]
        self.unet = unet
        self.noise_level = noise_level
        self.c = c

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        file_name = self.name_list[item]
        image_size = self.image_shape[1]
        # print(file_name)
        img = np.load(os.path.join(self.path,file_name))
        # print(img.shape, img.min(), img.max(), img.mean())
        # img = img[None, ...] * 1e3
        # print(img.shape)
        max_val = img.max()
        img = torch.tensor(img, dtype = torch.float32)
        img = F.interpolate(img[None,None,...], size = image_size,
                    mode='bilinear',
                    align_corners=True,
                    antialias=True)[0]
        
        shift_seed = img[0,image_size//2,image_size//2]
        torch.manual_seed(item + shift_seed)
        noise = torch.randn(self.image_shape) * self.noise_level * max_val
        
        # print(img.shape, noise.shape)
        y = img + noise

        if not self.unet:
            img = img.permute([1,2,0]).reshape(-1, self.c)

        return img, y






