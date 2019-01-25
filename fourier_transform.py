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

X_gridDIM = 512
X_amplitude = 5.

############################################################################

k = np.arange(X_gridDIM)
dX = 2 * X_amplitude / X_gridDIM

# the coordinate grid
X = (k - X_gridDIM / 2) * dX

############################################################################
#
#   plot the original
#
############################################################################

# randomly generate the width of the gaussian
alpha = np.random.uniform(1., 10.)

# randomly generate the displacement of the gaussian
a = np.random.uniform(0., 0.2 * X_amplitude)

# original function
f = np.exp(-alpha * (X - a) ** 2)

# the exact continious Fourier transform
FT_exact = lambda p: np.exp(-1j * a * p - p ** 2 / (4. * alpha)) * np.sqrt(np.pi / alpha)

plt.subplot(221)
plt.title('Original function')
plt.plot(X, f)
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
P = fftpack.fftfreq(X_gridDIM, dX / (2. * np.pi))

plt.subplot(222)
plt.title("Incorrect method")

plt.plot(P, FT_incorrect.real, label='real FFT')
plt.plot(P, FT_incorrect.imag, label='imag FFT')
plt.plot(P, FT_exact(P).real, label='real exact')
plt.plot(P, FT_exact(P).imag, label='imag exact')

plt.legend()
plt.xlabel('$p$')

############################################################################
#
#   correct method : Use the first method from
#   http://epubs.siam.org/doi/abs/10.1137/0915067
#
############################################################################
minus = (-1) ** k
FT_approx1 = dX * minus * fftpack.fft(minus * f, overwrite_x=True)

# get the corresponding momentum grid
P = (k - X_gridDIM / 2) * (np.pi / X_amplitude)

plt.subplot(223)
plt.title("Correct method #1 (using FFT)")
plt.plot(P, FT_approx1.real, label='real approximate')
plt.plot(P, FT_approx1.imag, label='imag approximate')
plt.plot(P, FT_exact(P).real, label='real exact')
plt.plot(P, FT_exact(P).imag, label='imag exact')
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
P_amplitude = 0.5 * alpha
dP = 2. * P_amplitude / X_gridDIM
P = (k - X_gridDIM / 2) * dP

delta = dX * dP / (2. * np.pi)

FT_approx2 = dX * np.exp(np.pi * 1j * (k - X_gridDIM / 2) * X_gridDIM * delta) * \
             frft(f * np.exp(np.pi * 1j * k * X_gridDIM * delta), delta)

plt.subplot(224)
plt.title("Correct method #2 (using FRFT)")
plt.plot(P, FT_approx2.real, label='real approximate')
plt.plot(P, FT_approx2.imag, label='imag approximate')
plt.plot(P, FT_exact(P).real, label='real exact')
plt.plot(P, FT_exact(P).imag, label='imag exact')
plt.legend()
plt.xlabel('$p$')

plt.show()





