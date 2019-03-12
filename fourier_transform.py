import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

__doc__ = """
Compare different ways of computing the continuous Fourier transform
"""

print(__doc__)

np.random.seed(5464)

############################################################################
#
#   parameters defining the coordinate grid
#
############################################################################

x_grid_dim = 512
x_amplitude = 5.

############################################################################

k = np.arange(x_grid_dim)
dx = 2 * x_amplitude / x_grid_dim

# the coordinate grid
x = (k - x_grid_dim / 2) * dx

############################################################################
#
#   plot the original
#
############################################################################

# randomly generate the width of the gaussian
alpha = np.random.uniform(1., 10.)

# randomly generate the displacement of the gaussian
a = np.random.uniform(0., 0.2 * x_amplitude)

# original function
f = np.exp(-alpha * (x - a) ** 2)

# the exact continious Fourier transform
ft_exact = lambda p: np.exp(-1j * a * p - p ** 2 / (4. * alpha)) * np.sqrt(np.pi / alpha)

plt.subplot(221)
plt.title('Original function')
plt.plot(x, f)
plt.xlabel('$x$')
plt.ylabel('$\\exp(-\\alpha x^2)$')

############################################################################
#
#   incorrect method: Naive
#
############################################################################

# Note that you may often see fftpack.fftshift used in this context

FT_incorrect = fftpack.fft(f)

# get the corresponding momentum grid
p = fftpack.fftfreq(x_grid_dim, dx / (2. * np.pi))

plt.subplot(222)
plt.title("Incorrect method")

plt.plot(p,FT_incorrect.real, label='real FFT')
plt.plot(p,FT_incorrect.imag, label='imag FFT')
plt.plot(p,ft_exact(p).real, label='real exact')
plt.plot(p,ft_exact(p).imag, label='imag exact')

plt.legend()
plt.xlabel('$p$')

############################################################################
#
#   correct method : Use the first method from
#   http://epubs.siam.org/doi/abs/10.1137/0915067
#
############################################################################
minus = (-1) ** k
ft_approx1 = dx * minus * fftpack.fft(minus * f, overwrite_x=True)

# get the corresponding momentum grid
p = (k - x_grid_dim / 2) * (np.pi / x_amplitude)

plt.subplot(223)
plt.title("Correct method #1 (using FFT)")
plt.plot(p, ft_approx1.real, label='real approximate')
plt.plot(p, ft_approx1.imag, label='imag approximate')
plt.plot(p, ft_exact(p).real, label='real exact')
plt.plot(p, ft_exact(p).imag, label='imag exact')
plt.legend()
plt.xlabel('$p$')

############################################################################
#
#   correct method : Use the second method from
#   http://epubs.siam.org/doi/abs/10.1137/0915067
#
############################################################################

def frft(x, alpha):
    """
    Implementation of the Fractional Fourier Transform (FRFT)
    :param x: array of data to be transformed
    :param alpha: parameter of FRFT
    :return: FRFT(x)
    """
    k = np.arange(x.size)

    y = np.hstack([
        x * np.exp(-np.pi * 1j * k**2 * alpha),
        np.zeros(x.size, dtype=np.complex)
    ])
    z = np.hstack([
        np.exp(np.pi * 1j * k**2 * alpha),
        np.exp(np.pi * 1j * (k - x.size)**2 * alpha)
    ])

    G = fftpack.ifft(
        fftpack.fft(y, overwrite_x=True) * fftpack.fft(z, overwrite_x=True),
        overwrite_x=True
    )

    return np.exp(-np.pi * 1j * k**2 * alpha) * G[:x.size]

# generate the desired momentum grid
p_amplitude = 1. * alpha

dp = 2. * p_amplitude / x_grid_dim
p = (k - x_grid_dim / 2) * dp

delta = dx * dp / (2. * np.pi)

ft_approx2 = dx * np.exp(np.pi * 1j * (k - x_grid_dim / 2) * x_grid_dim * delta) * \
             frft(f * np.exp(np.pi * 1j * k * x_grid_dim * delta), delta)

plt.subplot(224)
plt.title("Correct method #2 (using FRFT)")
plt.plot(p, ft_approx2.real, label='real approximate')
plt.plot(p, ft_approx2.imag, label='imag approximate')
plt.plot(p, ft_exact(p).real, label='real exact')
plt.plot(p, ft_exact(p).imag, label='imag exact')
plt.legend()
plt.xlabel('$p$')

plt.show()





