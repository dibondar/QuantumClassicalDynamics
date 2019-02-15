import numpy as np
from scipy.sparse import diags # Construct a sparse matrix from diagonals


class ForwardDiffQHamiltonian:
    """
    Construct the quantum Hamiltonian for an 1D system in the coordinate representation
    using the forward finite difference approximation.
    """
    def __init__(self, *, x_grid_dim, x_amplitude, v):
        """
         The following parameters must be specified
             x_grid_dim - the grid size
             x_amplitude - the maximum value of the coordinates
             v - a potential energy (as a function)
         """
        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.v = v

        # generate coordinate range
        self.x = np.linspace(-self.x_amplitude, self.x_amplitude, self.x_grid_dim)

        # save the coordinate step size
        self.dx = self.x[1] - self.x[0]

        # Construct the kinetic energy part as sparse matrix from diagonal
        self.hamiltonian = diags([1., -2., 1.], [0, 1, 2], shape=(self.x_grid_dim, self.x_grid_dim))
        self.hamiltonian *= -0.5 / (self.dx ** 2)

        # Add diagonal potential energy
        self.hamiltonian += diags(self.v(self.x), 0)
