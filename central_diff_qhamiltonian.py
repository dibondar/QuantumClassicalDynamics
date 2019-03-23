import numpy as np
from scipy.sparse import diags # Construct a sparse matrix from diagonals
from scipy.sparse import linalg # Linear algebra for sparse matrix


class CentralDiffQHamiltonian:
    """
    Construct the quantum Hamiltonian for an 1D system in the coordinate representation
    using the central finite difference approximation.
    """
    def __init__(self, *, x_grid_dim, x_amplitude, v, **kwargs):
        """
         The following parameters must be specified
             x_grid_dim - the grid size
             x_amplitude - the maximum value of the coordinates
             v - a potential energy (as a function)
             kwargs - is ignored
         """
        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.v = v

        # get coordinate step size
        self.dx = 2. * self.x_amplitude / self.x_grid_dim

        # generate coordinate range
        self.x = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * self.dx
        # The same as
        # self.x = np.linspace(-self.x_amplitude, self.x_amplitude - self.dx , self.x_grid_dim)

        # Construct the kinetic energy part as sparse matrix from diagonal
        self.hamiltonian = diags([1., -2., 1.], [-1, 0, 1], shape=(self.x_grid_dim, self.x_grid_dim))
        self.hamiltonian *= -0.5 / (self.dx ** 2)

        # Add diagonal potential energy
        self.hamiltonian += diags(self.v(self.x), 0)

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
            # using specialized function for sparse Hermitian matrices
            self.energies, self.eigenstates = linalg.eigsh(self.hamiltonian, which='SM', k=20)

            # transpose for convenience
            self.eigenstates = self.eigenstates.T

            # normalize each eigenvector
            for psi in self.eigenstates:
                psi /= np.linalg.norm(psi) * np.sqrt(self.dx)

            # Make sure that the ground state is non negative
            np.abs(self.eigenstates[0], out=self.eigenstates[0])

        return self