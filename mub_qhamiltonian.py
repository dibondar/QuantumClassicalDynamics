import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix


class MUBQHamiltonian(object):
    """
    Generate quantum Hamiltonian, H(x,p) = K(p) + V(x),
    for 1D system in the coordinate representation using mutually unbiased bases (MUB).
    """
    def __init__(self, *, x_grid_dim, x_amplitude, v, k, **kwargs):
        """
         The following parameters must be specified
             x_grid_dim - the grid size
             x_amplitude - the maximum value of the coordinates
             v - the potential energy (as a function)
             k - the kinetic energy (as a function)
             kwargs is ignored
         """
        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.v = v
        self.k = k

        # Check that all attributes were specified
        # make sure self.x_amplitude has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"

        # get coordinate step size
        self.dx = 2. * self.x_amplitude / self.x_grid_dim

        # generate coordinate range
        k = np.arange(self.x_grid_dim)
        self.x = (k - self.x_grid_dim / 2) * self.dx
        # The same as
        # self.x = np.linspace(-self.x_amplitude, self.x_amplitude - self.dx , self.x_grid_dim)

        # generate momentum range as it corresponds to FFT frequencies
        self.p = (k - self.x_grid_dim / 2) * (np.pi / self.x_amplitude)

        # 2D array of alternating signs
        minus = (-1) ** (k[:, np.newaxis] + k[np.newaxis, :])
        # see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # for explanation of np.newaxis and other array indexing operations
        # also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # for understanding the broadcasting in array operations    

        # Construct the momentum dependent part
        self.hamiltonian = np.diag(self.k(self.p))
        
        self.hamiltonian *= minus
        self.hamiltonian = fftpack.fft(self.hamiltonian, axis=1, overwrite_x=True)
        self.hamiltonian = fftpack.ifft(self.hamiltonian, axis=0, overwrite_x=True)
        self.hamiltonian *= minus

        # Add diagonal potential energy
        self.hamiltonian += np.diag(self.v(self.x))

    def get_eigenstate(self, n):
        """
        Return n-th eigenfunction
        :param n: order
        :return: a copy of numpy array containing eigenfunction
        """
        self.diagonalize()
        return self.eigenstates[n].copy()

    def get_energy(self, n):
        """
        Return the energy of the n-th eigenfunction
        :param n: order
        :return: real value
        """
        self.diagonalize()
        return self.energies[n]

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian if necessary
        :return: self
        """
        # check whether the hamiltonian has been diagonalized
        try:
            self.eigenstates
            self.energies
        except AttributeError:
            # eigenstates have not been calculated so
            # get real sorted energies and underlying wavefunctions
            # using specialized function for Hermitian matrices
            self.energies, self.eigenstates = linalg.eigh(self.hamiltonian)

            # extract real part of the energies
            self.energies = np.real(self.energies)

            # covert to the formal convenient for storage
            self.eigenstates = self.eigenstates.T

            # normalize each eigenvector
            for psi in self.eigenstates:
                psi /= linalg.norm(psi) * np.sqrt(self.dx)

            # Make sure that the ground state is non negative
            np.abs(self.eigenstates[0], out=self.eigenstates[0])

        return self