import torch
import numpy as np

def compute_fourier_kernel(N): 
    """Computes the Fourier space kernel which represents the mapping between
        convergence (kappa) and shear (gamma).
    Args:
        N (int): x,y dimension of image patch (assumes square images).
    Returns:
        𝒟 (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
    """
    # Generate grid of Fourier domain
    kx = torch.arange(N, dtype = torch.float32) - N/2
    kx, ky = torch.meshgrid(kx, kx, indexing='ij')
    k = kx**2+ky**2
    # Define Kaiser-Squires kernel
    D = torch.ones((N, N), dtype=torch.complex64)
    D = torch.where(k > 0, ((kx ** 2.0 - ky ** 2.0) + 1j * (2.0 * kx * ky))/k, D)
    # Apply inverse FFT shift
    D = torch.fft.ifftshift(D)
    return D



def forward_model(kappa, D):
    """Applies the forward mapping between convergence and shear through their
        relationship in Fourier space.
    Args:
        kappa (jnp.ndarray): Convergence field, with shape [N,N].
        D (jnp.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
    Returns:
        γ (jnp.ndarray): Shearing field, with shape [N,N].
    """
    f_kappa = torch.fft.fft2(kappa) # Perform 2D forward FFT
    f_y = f_kappa * D[None,...] # Map convergence onto shear
    return torch.fft.ifft2(f_y) # Perform 2D inverse FFT


def noise_maker(y, theta, ngrid, ngal):
    """Adds some random Gaussian noise to a mock weak lensing map.

    Args:
        theta (float): Opening angle in deg.
        ngrid (int): Number of grids.
        ngal (int): Number of galaxies.
        kappa (np.ndarray): Convergence map.
    
    Returns:
        gamma (np.ndarray): A synthetic representation of the shear field, gamma, with added noise.
    """
    sigma = 0.37 / np.sqrt(((theta*60/ngrid)**2)*ngal)
    noise = sigma*(torch.normal(mean = torch.zeros((y.shape[0],ngrid,ngrid), dtype = torch.float32),
                                std = 1) + 1j * torch.normal(mean = torch.zeros((y.shape[0],ngrid,ngrid), dtype = torch.float32),
                                std = 1))
    
    y = y + noise.to(y.device)
    return y


def inverse_model(y, D):
    """Applies the forward mapping between convergence and shear through their
        relationship in Fourier space.
    Args:
        kappa (jnp.ndarray): Convergence field, with shape [N,N].
        D (jnp.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
    Returns:
        γ (jnp.ndarray): Shearing field, with shape [N,N].
    """
    f_y = torch.fft.fft2(y) # Perform 2D forward FFT
    D_inv = (torch.conj(D) / (torch.square(torch.abs(D))))
    f_kappa = f_y * D_inv[None,...] # Map convergence onto shear
    # f_kappa[:,0,0] = 0
    return torch.fft.ifft2(f_kappa) # Perform 2D inverse FFT



def kaiser_squires(kappa, D):

    image_size = kappa.shape[1]
    y = forward_model(kappa, D)
    y = noise_maker(y, 5.0, image_size, 30)
    kappa_ks = inverse_model(y, D)
    kappa_ks = torch.cat((kappa_ks.real, kappa_ks.imag), dim = 0)

    return kappa_ks


def measurement(kappa, D):

    image_size = kappa.shape[1]
    y = forward_model(kappa, D)
    y = noise_maker(y, 5.0, image_size, 30)
    shear = torch.cat((y.unsqueeze(1).real, y.unsqueeze(1).imag), dim = 1)

    return shear

# torch.manual_seed(0)
# D = compute_fourier_kernel(1024)
# kappa = torch.rand(10, 1024, 1024)

# ys = forward_model(kappa, D)
# ys = noise_maker(ys, 5.0, 1024, 30)
# kappa_hat = inverse_model(ys, D)

# print(np.square(kappa_hat - kappa).sum())
