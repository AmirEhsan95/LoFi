import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy.ndimage import gaussian_filter


def load_M51():
    x_true = gaussian_filter(fits.getdata("M51.fits"), 1)
    return x_true

def random_sampling(samples, seed=562248):
    """random gaussian sampling pattern with sigma=pi/3 between -pi, pi"""
    np.random.seed(seed)
    uv = np.random.normal(size=(int(samples*1.05),2)) * np.pi/3 # generates more samples to make sure we have enough in (-pi,pi)
    sel = (abs(uv[:,0]) < np.pi) * (abs(uv[:,1]) < np.pi)
    return uv[sel][:samples]

# from src.data import load_M51
# from src.sampling.uv_sampling import random_sampling


import numpy as np
from scipy.special import iv, jv

def calculate_kaiser_bessel_coef(k, i, Jd=(6,6)):
    """Calculate the Kaiser-Bessel kernel coefficients for a 2d grid for the neighbouring pixels. 

    Args:
        k (float,float): location of the point to be interpolated
        i (int): extra index parameter
        Jd (tuple, optional): Amount of neighbouring pixels to be used in each direction. Defaults to (6,6).

    Returns:
        indices (list): list of indices of all the calculated coefficients
        values (list): list of the calculated coefficients
    """

    k = k.reshape(-1,1)
    J = Jd[0]//2
    a = np.array(np.meshgrid(range(-J, J), range(-J, J))).reshape(2, -1)
    a += (k % 1 >0.5) # corrects to the closest 6 pixels
    indices = (k.astype(int) + a)

    J = Jd[0]

    beta = 2.34*J
    norm = J 

    # for 2d do the interpolation 2 times, once in each direction
    u =  k.reshape(2,1) - indices
    values1 = iv(0, beta*np.sqrt(1 +0j - (2*u[0]/Jd[0])**2)).real / J 
    values2 = iv(0, beta*np.sqrt(1 +0j - (2*u[1]/Jd[0])**2)).real / J 
    values = values1 * values2
    
    indices = np.vstack((
            np.zeros(indices.shape[1]), 
            np.repeat(i, indices.shape[1]), indices[0], indices[1])
            ).astype(int)

    return indices.T, values


class NUFFT2D_Torch():
    """NUFFT implementation using a Kaiser-Bessel kernel for interpolation. 
    Implemented with TF operations. Only able to do the FFT on the last 2 axes 
    of the tensors provided. Slower than using the numpy_function on the np 
    based operations.
    """
    def __init__(self, device):
        self.device = device
        pass
        
    def plan(self, uv, Nd, Kd, Jd, batch_size):
        # saving some values
        self.Nd = Nd
        self.Kd = Kd
        self.Jd = Jd
        self.batch_size = batch_size
        self.n_measurements = len(uv)
        
        self.K_norm = max(Kd[0], Kd[1])
        gridsize = 2*np.pi / self.K_norm
        k = (uv + np.pi) / gridsize
        
        # calculating coefficients for interpolation
        indices = []
        values =  []
        for i in tqdm.tqdm(range(len(uv))):
            ind, vals = calculate_kaiser_bessel_coef(k[i], i, Jd)
            indices.append(ind)
            values.append(vals.real)

        values = np.array(values).reshape(-1)
        indices = np.array(indices).reshape(-1, 4)

        self.indices = indices
        # check if indices are within bounds, otherwise suppress them and raise warning
        if np.any(indices[:,2:] < 0) or np.any(indices[:,2] >= Kd[0]) or np.any(indices[:,3] >= Kd[1]):
            sel_out_bounds = np.any([np.any(indices[:,2:] < 0, axis=1), indices[:,2] >= Kd[0], indices[:,3] >= Kd[1]], axis=0)
            print(f"some values lie out of the interpolation array, these are not used, check baselines")
            indices[sel_out_bounds] = 0
            values[sel_out_bounds] = 0
        
        # repeating the values and indices to match the batch_size 
        batch_indices = np.tile(indices[:,-2:], [batch_size, 1])
        batch_indicators = np.repeat(np.arange(batch_size), (len(values)))
        batch_indices = np.hstack((batch_indicators[:,None], batch_indices))
        
        self.flat_batch_indices = torch.LongTensor(np.ravel_multi_index(batch_indices.T, (batch_size, Kd[0], Kd[1]))).to(self.device)
        
        self.batch_indices = list(torch.LongTensor(batch_indices).to(self.device).T)
        batch_values = np.tile(values, [batch_size,1]).astype(np.float32).reshape(self.batch_size, self.n_measurements, self.Jd[0]*self.Jd[1])
        self.batch_values = torch.tensor(batch_values, dtype=torch.float32).to(self.device)


        # determine scaling based on iFT of the KB kernel
        J = Jd[0] 
        beta = 2.34*J
        s_kb = lambda x: np.sinc(np.sqrt((np.pi *x *J)**2 - (2.34*J)**2 +0j)/np.pi)

        xx_1 = (np.arange(Kd[0])/Kd[0] -.5)[(Kd[0]-Nd[0])//2:(Kd[0]-Nd[0])//2 + Nd[0]]
        xx_2 = (np.arange(Kd[1])/Kd[1] -.5)[(Kd[1]-Nd[1])//2:(Kd[1]-Nd[1])//2 + Nd[1]]
        
        sa_1 = s_kb(xx_1).real
        sa_2 = s_kb(xx_2).real
        
        self.scaling = (sa_1.reshape(-1,1) * sa_2.reshape(1,-1)).reshape(1, Nd[0], Nd[1])
        self.scaling = torch.tensor(self.scaling, dtype=torch.complex64).to(self.device)
        self.forward = self.dir_op
        self.adjoint = self.adj_op


    def dir_op(self, xx):
        # xx = torch.tensor(xx, dtype=torch.complex64)
        xx= xx[0].to(torch.complex64)
        xx = xx/self.scaling
        xx = self._pad(xx)
        kk = self._xx2kk(xx) / self.K_norm
        k = self._kk2k(kk)
        k = k[None,...]
        return k


    def adj_op(self, k):
        k= k[0]
        # split real and imaginary parts because complex operations not defined for sparseTensors

        kk = self._k2kk(k)
        xx = self._kk2xx(kk) * self.K_norm
        xx = self._unpad(xx)
        xx = xx / self.scaling
        xx = xx[None,...]
        return xx.real
    
    def _kk2k(self, kk):
        """interpolates of the grid to non uniform measurements"""
        
        return (kk[self.batch_indices].reshape(self.batch_size, self.n_measurements, self.Jd[0]*self.Jd[1]) * self.batch_values).sum(axis=-1)
    
    def _k2kk(self, k):
        """convolutes measurements to oversampled fft grid"""
        interp = (k.reshape(self.batch_size, self.n_measurements, 1) * self.batch_values).reshape(-1)
        
        kk_flat = torch.zeros(self.batch_size * self.Kd[0] * self.Kd[1], dtype=torch.complex64).to(self.device)
        kk_flat.scatter_add_(0, self.flat_batch_indices, interp )

        return kk_flat.reshape(self.batch_size, self.Kd[0], self.Kd[1])
    
    @staticmethod
    def _kk2xx(kk):
        """from 2d fourier space to 2d image space"""
        return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(kk, dim=(-2,-1))), dim=(-2,-1)) 

    @staticmethod
    def _xx2kk(xx):
        """from 2d fourier space to 2d image space"""
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(xx, dim=(-2,-1))), dim=(-2,-1))
    
    def _pad(self, x):
        """pads x to go from Nd to Kd"""
        return torch.nn.functional.pad(x, (
            (self.Kd[1]-self.Nd[1])//2, (self.Kd[1]-self.Nd[1])//2,
            (self.Kd[0]-self.Nd[0])//2, (self.Kd[0]-self.Nd[0])//2,
            0, 0,
            )
        )
    
    def _unpad(self, x):
        """unpads x to go from  Kd to Nd"""
        return x[
            :,
            (self.Kd[0]-self.Nd[0])//2: (self.Kd[0]-self.Nd[0])//2 +self.Nd[0],
            (self.Kd[1]-self.Nd[1])//2: (self.Kd[1]-self.Nd[1])//2 +self.Nd[1]
            ] 
    




class KbNuFFT2d_torch(torch.nn.Module):
    """Alternative implementation of the NuFFT with Kaisser-Besel kernels.

    Parameters
    ----------
    uv : torch.Tensor
        uv plane with N measurements. Shape (2, N)
    im_size : tuple
        Size of image to reconstruct. Shape (n, m)
    device : torch.device
        Torch device.
    interp_points : Union[int, Sequence[int]]
        Number of neighbors to use for interpolation in each dimension.
    k_oversampling :  Union[int, float]
        Oversampling of the k space grid, should be between `1.25` and `2`. Usually set to `2`.
    norm_type : str
        Whether to apply normalization with the FFT operation. Options are ``"ortho"`` or ``None``.
    myType : torch.dtype
        Type for float numbers.
    myComplexType : torch.dtype
        Type for complex numbers.

    """

    def __init__(
        self,
        uv,
        im_shape,
        device,
        interp_points=7,
        k_oversampling=2,
        norm_type="ortho",
        myType=torch.float32,
        myComplexType=torch.complex64,
    ):
        super().__init__()
        assert len(uv.shape) == 2
        assert len(im_shape) == 2
        self.uv = uv
        self.im_shape = im_shape
        self.interp_points = interp_points
        self.myType = myType
        self.myComplexType = myComplexType
        self.norm_type = norm_type
        self.device = device

        # Define oversampled grid
        self.grid_size = (
            int(self.im_shape[0] * k_oversampling),
            int(self.im_shape[1] * k_oversampling),
        )
        # To be computed
        self.norm = None

        # Init interpolation matrix
        self.init_interp_matrix()

        # Initialise base operator layers
        self.forwardOp = tkbn.KbNufft(
            im_size=self.im_shape,
            grid_size=self.grid_size,
            numpoints=self.interp_points,
            device=self.device,
            dtype=self.myType,
        )
        self.adjointOp = tkbn.KbNufftAdjoint(
            im_size=self.im_shape,
            grid_size=self.grid_size,
            numpoints=self.interp_points,
            device=self.device,
            dtype=self.myType,
        )
        # Compute norm
        self.compute_norm()

    def init_interp_matrix(self):
        with torch.no_grad():
            self.interp_mat = tkbn.calc_tensor_spmatrix(
                self.uv,
                im_size=self.im_shape,
                grid_size=self.grid_size,
                numpoints=self.interp_points,
            )

    def compute_norm(self):
        """Compute operator norm"""
        self.norm = max_eigenval(
            A=self.dir_op,
            At=self.adj_op,
            im_shape=(1, 1) + self.im_shape,
            tol=1e-4,
            max_iter=np.int64(1e4),
            verbose=0,
            device=self.device,
        )

    def dir_op(self, x):
        """Forward operator.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, H, W), as the channel dimension C should be 1.
        """
        return self.forwardOp(
            image=x.to(self.myComplexType),
            omega=self.uv,
            interp_mats=self.interp_mat,
            norm=self.norm_type,
        )

    def adj_op(self, k):
        """Adjoint operator.

        Parameters
        ----------
        k : torch.Tensor
            Measurement set corresponding to the stored uv plane. Shape (B, 1, N).
        """
        return self.adjointOp(
            data=k, omega=self.uv, interp_mats=self.interp_mat, norm=self.norm_type
        )



def compute_complex_sigma_noise(observations, input_snr):
    """Computes the standard deviation of a complex Gaussian noise

    The eff_sigma is such that `Im(n), Re(n) \sim N(0,eff_sigma)`, where
    eff_sigma=sigma/sqrt(2)

    Args:
        observations (np.ndarray): complex observations
        input_snr (float): desired input SNR

    Returns:
        eff_sigma (float): effective standard deviation for the complex Gaussian noise
    """
    num_measurements = observations[observations != 0].shape[0]

    sigma = 10 ** (-input_snr / 20) * (
        np.linalg.norm(observations.flatten(), ord=2) / np.sqrt(num_measurements)
    )
    eff_sigma = sigma / np.sqrt(2)

    return eff_sigma
