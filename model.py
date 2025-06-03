import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class FourierFeatures(nn.Module):
    """Random Fourier features.
    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """
    def __init__(self, frequency_matrix, learnable_features=False, scale_net = None):
        super(FourierFeatures, self).__init__()
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)


        self.scale_net = scale_net


        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates, z=None):
        """Creates Fourier features from coordinates.
        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)

        if self.scale_net is not None:
            scale = self.scale_net(z)[...,None]
            prefeatures = prefeatures * scale
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * torch.pi * prefeatures)
        sin_features = torch.sin(2 * torch.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=1)


class MLP_net(nn.Module):
    def __init__(self, prev_unit, out_unit, num_layers = 3, hidden_dim = 128):
        super(MLP_net, self).__init__()
        hidden_units = [hidden_dim,] * num_layers + [out_unit]
        fcs = []
        for i in range(len(hidden_units)):
            fcs.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]

        self.fcs = nn.ModuleList(fcs)

    def forward(self, x):
        for i in range(len(self.fcs)-1):
            x = F.relu(self.fcs[i](x))
        x = self.fcs[-1](x)

        return x
    


class LoFi(nn.Module):
    '''LoFi module'''

    def __init__(self, image_size = 128, c_in = 3, c_out = 3, network = 'MLP',
                 hidden_dim = 256, num_layers = 3,
                 patch_shape = 'round', fourier_filtering = True,
                 recep_scale = 1, residual_learning = False,
                 learned_geo = True, N = 9, M = 9, CCPG = False,
                 num_filters = 1, n_deform = 1, coord_deform = False):
        super(LoFi, self).__init__()

        self.image_size = image_size
        self.c_in = c_in
        self.c_out = c_out
        self.network = network
        self.fourier_filtering = fourier_filtering
        self.recep_scale = recep_scale
        self.residual_learning = residual_learning
        self.patch_shape = patch_shape
        self.learned_geo = learned_geo
        self.N = N
        self.M = M
        self.CCPG = CCPG
        self.num_filters = num_filters
        self.n_deform = n_deform
        self.coord_deform = coord_deform
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        prev_unit = self.N * self.M * self.c_in


        if self.fourier_filtering:

            if not self.c_in == 2:
                prev_unit = self.N * self.M * self.c_in * 3
            
            else:
                prev_unit = self.N * self.M * self.c_in * 2


        if self.CCPG:

            MLP_deforms = []
            for _ in range(self.n_deform):
                MLP_deforms.append(MLP_net(prev_unit=prev_unit,
                                    out_unit=self.N * self.M*2,
                                    inter_unit = 128))
            
            self.MLP_deforms = nn.ModuleList(MLP_deforms)
            init_deform_weight = torch.zeros(self.n_deform)
            self.deform_weights = nn.Parameter(init_deform_weight.clone().detach(), requires_grad=True)
            init_uniform_weight = torch.ones(self.n_deform)
            self.uniform_weights = nn.Parameter(init_uniform_weight.clone().detach(), requires_grad=True)

        
        if self.coord_deform:

            self.INR = MLP_net(prev_unit = 2, out_unit= 2, inter_unit = 256)
            

        if self.network == 'MultiMLP':
            total_features = self.M * 100
            num_mlps = self.M
            input_dim = prev_unit//self.M
            fcs = []
            for _ in range(num_mlps):
                fcs.append(MLP_net(input_dim, total_features//num_mlps, num_layers = num_layers, hidden_dim = hidden_dim))


            self.mixer_MLP = MLP_net(total_features, self.c_out, num_layers = num_layers, hidden_dim = hidden_dim)
            self.MLP = nn.ModuleList(fcs)
        
        elif self.network == 'MLP':

            fcs = MLP_net(prev_unit, self.c_out, num_layers=num_layers, hidden_dim= hidden_dim)
            self.MLP = nn.ModuleList(fcs)


        if self.fourier_filtering:
            self.pad_size = 5
            c_f = 1 if self.c_in == 2 else self.num_filters * self.c_in 
            filter = torch.ones(1,c_f, self.image_size + 2*self.pad_size,
                                self.image_size + 2*self.pad_size, dtype = torch.complex64)
            self.filter = nn.Parameter(filter.clone().detach(), requires_grad=True)

        

        if self.patch_shape == 'round':

            r = self.N/self.image_size
            thetas = torch.arange(self.M)*(2*np.pi/self.M)
            x = r*torch.cos(thetas)/(2*self.N)
            y = r*torch.sin(thetas)/(2*self.N)
            x = x[...,None]
            y = y[...,None]
            xy = torch.concat([x,y], dim = 1)[None,...]
            xy = xy.expand(self.N,-1,-1)
            idx = (torch.arange(0,self.N))[...,None,None]
            patch = idx * xy

        elif self.patch_shape == 'square':

            x = torch.arange(-(self.N//2), self.N//2+1)/(self.image_size)
            y = torch.arange(-(self.M//2), self.M//2+1)/(self.image_size)
            x , y = torch.meshgrid(x,y, indexing='ij')
            x = x[...,None]
            y = y[...,None]
            patch = torch.concat([x,y], dim = 2)[None,...]

        elif self.patch_shape == 'random':

            patch = 2 * self.N*(torch.rand(self.N, self.M,2) - 0.5)/(self.image_size)


        self.patch = nn.Parameter(patch.clone().detach(), requires_grad=self.learned_geo)
        # Adaptive receptive field
        patch_scale = self.recep_scale*torch.ones(1)
        self.patch_scale = nn.Parameter(patch_scale.clone().detach(), requires_grad=True) 

            

    def cropper(self, image, coordinate, patch, patch_scale,
                patch_analysis = False):
        '''Patch Extraction'''
        # Coordinate shape: b X b_pixels X 2
        # image shape: b X c X h X w
        b , c , h , w = image.shape
        b_pixels = coordinate.shape[1]
        coordinate = coordinate * 2

        patch = patch_scale * patch / (h/self.image_size)

        patch = patch[None, None]
        N = self.N
        M = self.M

        coordinate = coordinate.unsqueeze(2).unsqueeze(2)
        f = coordinate + patch

        if patch_analysis:
            return f

        f = f.reshape(b, b_pixels * N, M,2).flip(dims = [-1])

        image_cropped = F.grid_sample(image, f,
                                      mode = 'bicubic',
                                      align_corners=True,
                                      padding_mode='reflection')

        image_cropped = image_cropped.permute(0,2,3,1)
        image_cropped = image_cropped.reshape(b, b_pixels , N, M,c)
        image_cropped = image_cropped.reshape(b* b_pixels , N, M,c)
        image_cropped = image_cropped.permute(0,3,1,2)

        return image_cropped



    def noise_suppresion_filter(self, x):

        pad = [self.pad_size,self.pad_size,
            self.pad_size,self.pad_size]
        x_filtered = F.pad(x, pad, "constant", 0)

        if x.shape[1] == 2:
            x_filtered = torch.complex(x_filtered[:,0:1] , x_filtered[:,1:2])

        x_filtered = torch.fft.fft2(x_filtered, norm = 'forward')

        if not self.filter.shape[2] == x_filtered.shape[2]:
            filter_inter_real = F.interpolate(self.filter.real,
                                        size = x_filtered.shape[2],
                                        mode= 'bicubic')
            filter_inter_imag = F.interpolate(self.filter.imag,
                                        size = x_filtered.shape[2],
                                        mode= 'bicubic')
            filter_inter = torch.complex(filter_inter_real, filter_inter_imag)

        else:

            filter_inter = self.filter

        # print(x_filtered.shape, filter_inter.shape)
        x_filtered = x_filtered.repeat(1,filter_inter.shape[1]//x_filtered.shape[1], 1, 1)
        # print(x_filtered.shape, filter_inter.shape)
        x_filtered = x_filtered * filter_inter
        h_p = x_filtered.shape[2]
        x_filtered = torch.fft.ifft2(x_filtered, norm = 'forward')[:,:,h_p//2 - x.shape[2]//2:h_p//2 + x.shape[2]//2,
                                                h_p//2 - x.shape[2]//2:h_p//2 + x.shape[2]//2]
        
        x_filtered = torch.concat((x_filtered.real, x_filtered.imag), dim = 1)
        x = torch.concat((x_filtered, x), dim = 1)
            
        return x



    def forward(self, coordinate, x, patch_analysis = False):

        b , b_pixels , _ = coordinate.shape
        if self.fourier_filtering:
            x = self.noise_suppresion_filter(x)
        x_cropped = self.cropper(x , coordinate, self.patch, self.patch_scale)
        # x_cropped = self.stochastic_cropper(x , coordinate)

        if self.CCPG:

            for df in range(self.n_deform):
                x_cropped = torch.flatten(x_cropped,1)
                deformed_patch = self.MLP_deforms[df](x_cropped)
                deformed_patch = deformed_patch.reshape(b, b_pixels, self.N, self.M,2)
                patch_analysis_i = patch_analysis if df == self.n_deform-1 else False
                x_cropped = self.cropper(x,coordinate, deformed_patch, self.patch_scale, patch_analysis_i)

        
        if self.coord_deform:

            deformed_coord = self.INR(coordinate)
            x_cropped = self.cropper(x,deformed_coord, self.patch, self.patch_scale,
                                    patch_analysis= patch_analysis)
            

        if patch_analysis:
            return x_cropped

        if self.patch_shape == 'square':
            mid_pix = x_cropped[:,:self.c_out,4,4] # Centeric pixel
        elif self.patch_shape == 'round':
            mid_pix = x_cropped[:,:self.c_out,0,0] # Centeric pixel
        

        x = x_cropped

        if self.network == 'MultiMLP':

            chunk_outs = []
            for i in range(len(self.MLP)):
                chunk = x[:, :, :, i]
                chunk = torch.flatten(chunk,1)
                chunk_out = self.MLP[i](chunk)
                chunk_outs.append(chunk_out)
            
            x = torch.cat(chunk_outs, dim=1)
            x = self.mixer_MLP(x)

        elif self.network == 'MLP':
            x = torch.flatten(x,1)
            x = self.MLP(x) 
                    
        if self.residual_learning:
            x = mid_pix - x 
        x = x.reshape(b, b_pixels, -1)

        return x
    





