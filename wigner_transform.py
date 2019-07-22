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
    W = rho.copy()

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

psi_1 = np.exp(-(x + 4) ** 2)
psi_2 = np.exp(-(x - 2) ** 2)

schrodinger_cat =  psi_1 + psi_2 + 0j
schrodinger_cat /= np.linalg.norm(schrodinger_cat)
rho_averaged_initial = schrodinger_cat * schrodinger_cat.conj().T
schrodinger_cat = schrodinger_cat.reshape(-1)

########################################################################################################################
#
#   Adding noise and averaging over it to induce incoherence
#
########################################################################################################################

# the array to save the averaged density matrix over noise
rho_averaged_final = np.zeros_like(rho_averaged_initial)


from split_op_schrodinger1D import SplitOpSchrodinger1D, njit

np.random.seed(45)

# loop over noise
for index in range(100):

    F = np.random.uniform()

    psi = SplitOpSchrodinger1D(

        x_grid_dim=x_grid_dim,
        x_amplitude=x_amplitude,

        v=njit(lambda x, t: -4. * F * x),
        k=njit(lambda p, t: 0.5 * p ** 2),
        dt=0.05
    ).set_wavefunction(schrodinger_cat).propagate(10).reshape(x.shape)

    # form the density matrix out of the wavefunctions
    psi /= np.linalg.norm(psi)
    rho = psi * psi.conj().T

    # Calculate the iterative mean following http://www.heikohoffmann.de/htmlthesis/node134.html
    rho -= rho_averaged_final
    rho /= (index + 1)
    rho_averaged_final += rho

    print(index)

extent = [x_wigner.min(), x_wigner.max(), p_wigner.min(), p_wigner.max()]

imag_params = dict(
    extent=extent,
    origin='lower',
    cmap='seismic',
    aspect=(extent[1] - extent[0]) / (extent[-1] - extent[-2]),
    #norm=WignerNormalize(vmin=-0.1, vmax=0.1)
    norm=WignerNormalize()
)

plt.subplot(121)

plt.title("Initial condition $Tr(\hat\\rho^2) = {:.3f}$".format(
    rho_averaged_initial.dot(rho_averaged_initial).trace().real
))

plt.imshow(rho2wigner(rho_averaged_initial).real, **imag_params)
plt.colorbar()

plt.xlabel('$x$')
plt.ylabel('$p$')


plt.subplot(122)

plt.title("Initial condition $Tr(\hat\\rho^2) = {:.3f}$".format(
    rho_averaged_final.dot(rho_averaged_final).trace().real
))

plt.imshow(rho2wigner(rho_averaged_final).real, **imag_params)
plt.colorbar()

plt.xlabel('$x$')
plt.ylabel('$p$')

plt.show()