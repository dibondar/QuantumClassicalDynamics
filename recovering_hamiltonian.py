import numpy as np
from scipy import fftpack, linalg
from scipy.signal import find_peaks, blackman


class RecoverHamiltonian:

    def __init__(self, wavefunctions, dt=1., threshold=1e-3, **kwargs):
        """
        Recover the time-independent Hamiltonian from the time evolution of the wavefunction
        :param wavefunctions: time evolution saved as a list of array or 2D array,
            where the zero index corresponds to time amd the first -- basis.
            It is assumed the wavefunction represented in the orthogonal basis
        :param dt: time step (float)
        :param threshold: threshold for recovering the eigen energies by using sicpy.find_peaks.
            Note that this arguments is passed as height in sicpy.find_peaks
        :param kwargs: all other key-value arguments for function sicpy.find_peaks
        """

        wavefunctions = np.array(wavefunctions, copy=True)

        # normalize the wavefunction over the first axis (basis)
        wavefunctions /= linalg.norm(wavefunctions, axis=1)[:, np.newaxis]

        # calculate the alternating sequence of signs for iFFT autocorrelation function
        k = np.arange(wavefunctions.shape[0])
        minus = (-1) ** k[:, np.newaxis]

        # energy axis (as prescribed by Method 1 for calculating Fourier transform
        energy_range = (k - k.size / 2) * np.pi / (0.5 * dt * k.size)

        # the windowed fft of the wave function with respect to the time axis
        wavefunctions_fft_w = fftpack.ifft(
            minus * wavefunctions * blackman(k.size)[:, np.newaxis],
            axis=0,
            overwrite_x=True
        )
        wavefunctions_fft_w *= minus


        weight = linalg.norm(wavefunctions_fft_w, axis=1)
        weight /= weight.max()

        # extract peaks in weight to get the eigen energies
        peaks, _ = find_peaks(weight, height=threshold, **kwargs)
        #peaks = np.nonzero(weight > threshold)

        # the eigenvalues of the Hamiltonian
        energies = energy_range[peaks]

        self.weight = weight
        self.energy_range = energy_range

        """
        #############################################################################################


        # normalize the auto correlation function
        auto_corr_fft_w /= auto_corr_fft_w.max()

        # extract peaks in the auto correlation function to get the eigen energies
        peaks, _ = find_peaks(auto_corr_fft_w, height=threshold, **kwargs)

        # the eigenvalues of the Hamiltonian
        energies = energy_range[peaks]

        #############################################################################################

        # calculate the alternating sequence of signs for iFFT wavefunction
        minus = (-1) ** k[:, np.newaxis]

        wavefunctions_fft_w = fftpack.ifft(
            minus * wavefunctions * blackman(k.size)[:, np.newaxis],
            axis=0,
            overwrite_x=True
        )
        wavefunctions_fft_w *= minus
        """

        # extract the eigenfunctions of the unknown hamiltonian
        eigenvects = wavefunctions_fft_w[peaks]

        # normalize the eigenfunctions
        eigenvects /= linalg.norm(eigenvects, axis=1)[:, np.newaxis]

        # remove the numerical noise by orthogonalizing the extracted basis
        # This is a numerically stable version of the Gramm Schmidt
        eigenvects, _ = linalg.qr(eigenvects.T, mode='economic', overwrite_a=True)
        eigenvects = eigenvects.T

        # saving the results of recovering
        self.energies = energies
        self.eigenvects = eigenvects

        # save the initial condition
        self.init_wavefunc = wavefunctions[0].copy()

    def propagate(self, times):
        """
        Return the time evolution for the specified time steps in times
        :param times: numpy.array -- time range for which to save the time evolution.
        :return: 2D numpy.array
        """

        # get expansion coefficients for the initial state
        coefs = self.eigenvects.conj() @ self.init_wavefunc

        #
        tmp = (self.eigenvects * coefs[:, np.newaxis])

        propagated_states = np.sum(
            tmp[:, :, np.newaxis] * np.exp(
                -1j * self.energies[:, np.newaxis, np.newaxis] * times[np.newaxis, np.newaxis, :]),
            axis=0
        )

        return propagated_states.T