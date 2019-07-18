import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from wigner_normalize import WignerNormalize, WignerSymLogNorm

########################################################################################################################
#
#   parameters for the quantum system
#
########################################################################################################################

# grid size
x_grid_dim = 256

# amplitudes
x_amplitude = 10.

########################################################################################################################
#
#   Generating grids
#
########################################################################################################################

dx = 2. * x_amplitude / x_grid_dim

n = np.arange(x_grid_dim)[np.newaxis, :]

x = (n - x_grid_dim / 2.) * dx
p = (n - x_grid_dim / 2) * (np.pi / x_amplitude)

x_prime = x.T
p_prime = p.T

# grids for the wigner function
x_wigner = x / np.sqrt(2.)
p_wigner = p_prime / np.sqrt(2.)

########################################################################################################################
#
#   Declare functions for the direct and inverse Wigner transforms
#
########################################################################################################################

# pre-calculate phases for efficient calculations
minus_n_k = (-1) ** (n + n.T)
phase_shear_x = np.exp(-1j * p * x_prime * (np.sqrt(2.) - 1.))
phase_shear_y = np.exp(1j * x * p_prime / np.sqrt(2.))


def rho2wigner(rho):
    """
    The Wigner transform of the density matrix
    :param rho: the density matrix
    :return: numpy.array
    """
    # just change the name
    W = rho

    W *= minus_n_k

    W = fft(W, axis=1, overwrite_x=True)
    W *= phase_shear_x
    W = ifft(W, axis=1, overwrite_x=True)

    W = fft(W, axis=0, overwrite_x=True)
    W *= phase_shear_y
    W = ifft(W, axis=0, overwrite_x=True)

    W = fft(W, axis=1, overwrite_x=True)
    W *= phase_shear_x
    W = ifft(W, axis=1, overwrite_x=True)

    W = ifft(W, axis=0, overwrite_x=True)

    W *= minus_n_k

    # enforce normalization
    W /= W.sum() * np.pi / W.shape[0]

    # perform the checks
    assert np.linalg.norm(W.imag.reshape(-1), np.infty), "there should be no imaginary part"
    assert (W.real <= 2.).all() and (W.real > -2.).all(), "The Cauchy-Schwarz inequality is violated"

    return W

########################################################################################################################
#
#   Test example
#
########################################################################################################################


plt.subplot(121)
plt.title("Original Schrodinger cat")

# get the state
#psi_1 = np.exp(-(x) ** 2 +10j * x)
psi_1 = np.exp(-(x - 4) ** 2)
psi_2 = np.exp(-(x + 4) ** 2)

schrodinger_cat = psi_1 + psi_2

# get the Wigner function
W = rho2wigner(schrodinger_cat * schrodinger_cat.conj().T)

extent = [x_wigner.min(), x_wigner.max(), p_wigner.min(), p_wigner.max()]

imag_params = dict(
    extent=extent,
    origin='lower',
    cmap='seismic',
    aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
    norm=WignerNormalize(vmin=-0.1, vmax=0.1)
    #norm=WignerNormalize()
)

plt.imshow(W.real, **imag_params)
plt.colorbar()

plt.xlabel('$x$')
plt.ylabel('$p$')

########################################################################################################################
#
#   Adding noise and averaging over it to induce incoherence
#
########################################################################################################################

# the array to save the averaged density matrix over noise
rho_averaged = np.zeros_like(W)

# loop over noise
for index in range(1000):

    psi = psi_1 + psi_2 * np.exp(+1j * np.random.normal(loc=0, scale=2.8))
    psi /= np.linalg.norm(psi)

    # form the density matrix out of the wavefunctions
    rho = psi * psi.conj().T

    # Calculate the iterative mean following http://www.heikohoffmann.de/htmlthesis/node134.html
    rho -= rho_averaged
    rho /= (index + 1)
    rho_averaged += rho

# print(rho_averaged.dot(rho_averaged).trace())

plt.subplot(122)

plt.imshow(rho2wigner(rho_averaged).real, **imag_params)
plt.colorbar()

plt.xlabel('$x$')
plt.ylabel('$p$')

plt.show()