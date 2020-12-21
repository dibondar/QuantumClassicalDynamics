import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from numba import njit
from numba.core.registry import CPUDispatcher
from types import FunctionType

class SplitOpSchrodinger2D(object):
    """
    The second-order split-operator propagator of the 2D Schrodinger equation in the coordinate representation
    with the time-dependent Hamiltonian H = K(P1, P2, t) + V(X1, X2, t).
    """
    def __init__(self, x1_grid_dim, x2_grid_dim, x1_amplitude, x2_amplitude, v, k, dt, diff_k_p1=None, diff_k_p2=None,
                 diff_v_x1=None, diff_v_x2=None, t=0., abs_boundary=1., **kwargs):
        """
        :param x1_grid_dim: the x1 grid size
        :param x2_grid_dim: the x2 grid size

        :param x1_amplitude: the maximum value of the x1 coordinates
        :param x2_amplitude: the maximum value of the x2 coordinates

        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)

        :param diff_k_p1: the derivative of the kinetic energy w.r.t. p1 for the Ehrenfest theorem calculations
        :param diff_k_p2: the derivative of the kinetic energy w.r.t. p2 for the Ehrenfest theorem calculations

        :param diff_v_x1: the derivative of the potential energy w.r.t. x1 for the Ehrenfest theorem calculations
        :param diff_v_x2: the derivative of the potential energy w.r.t. x2 for the Ehrenfest theorem calculations

        :param t: initial value of time
        :param dt: time increment
        :param abs_boundary: absorbing boundary
        :param kwargs: ignored
        """

        # save the parameters
        self.x1_grid_dim = x1_grid_dim
        self.x2_grid_dim = x2_grid_dim
        self.x1_amplitude = x1_amplitude
        self.x2_amplitude = x2_amplitude
        self.diff_v_x1 = diff_v_x1
        self.diff_v_x2 = diff_v_x2
        self.diff_k_p1 = diff_k_p1
        self.diff_k_p2 = diff_k_p2
        self.dt = dt
        self.t = t
        self.abs_boundary = abs_boundary

        assert 2 ** int(np.log2(self.x1_grid_dim)) == self.x1_grid_dim and \
               2 ** int(np.log2(self.x2_grid_dim)) == self.x2_grid_dim, \
                "The grid size (x1_grid_dim and x2_grid_dim) must be a power of 2"


        # get coordinate step sizes
        self.dx1 = 2. * self.x1_amplitude / self.x1_grid_dim
        self.dx2 = 2. * self.x2_amplitude / self.x2_grid_dim

        # generate coordinate ranges
        k1 = np.arange(self.x1_grid_dim)[:, np.newaxis]
        k2 =  np.arange(self.x2_grid_dim)[np.newaxis, :]
        # see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # for explanation of np.newaxis and other array indexing operations
        # also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # for understanding the broadcasting in array operations

        x1 = self.x1 = (k1 - self.x1_grid_dim / 2) * self.dx1
        x2 = self.x2 = (k2 - self.x2_grid_dim / 2) * self.dx2

        # generate momentum ranges
        p1 = self.p1 = (k1 - self.x1_grid_dim / 2) * (np.pi / self.x1_amplitude)
        p2 = self.p2 = (k2 - self.x2_grid_dim / 2) * (np.pi / self.x2_amplitude)

        # allocate the array for wavefunction
        self.wavefunction = np.zeros((self.x1_grid_dim, self.x2_grid_dim), dtype=np.complex)

        ###################################################################################################
        #
        # Codes for efficient evaluation
        #
        ####################################################################################################

        if isinstance(abs_boundary, CPUDispatcher):
            @njit
            def expV(wavefunction, t):
                """
                function to efficiently evaluate
                    wavefunction *= (-1) ** (k1 + k2) * exp(-0.5j * dt * v)
                """
                wavefunction *= (-1) ** (k1 + k2) * abs_boundary(x1, x2) * np.exp(-0.5j * dt * v(x1, x2, t))

        elif isinstance(abs_boundary, (float, int)):
            @njit
            def expV(wavefunction, t):
                """
                function to efficiently evaluate
                    wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
                """
                wavefunction *= (-1) ** (k1 + k2) * abs_boundary * np.exp(-0.5j * dt * v(x1, x2, t))

        else:
            raise ValueError("abs_boundary must be either a numba function or a numerical constant")

        self.expV = expV

        @njit
        def expK(wavefunction, t):
            """
            function to efficiently evaluate
                wavefunction *= exp(-1j * dt * k)
            """
            wavefunction *= np.exp(-1j * dt * k(p1, p2, t))

        self.expK = expK

        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        if diff_k_p1 and diff_k_p2 and diff_v_x1 and diff_v_x2:

            # Get codes for efficiently calculating the Ehrenfest relations
            @njit
            def get_p1_average_rhs(density, t):
                return np.sum(density * diff_v_x1(x1, x2, t))

            self.get_p1_average_rhs = get_p1_average_rhs

            @njit
            def get_p2_average_rhs(density, t):
                return np.sum(density * diff_v_x2(x1, x2, t))

            self.get_p2_average_rhs = get_p2_average_rhs

            @njit
            def get_v_average(density, t):
                return np.sum(v(x1, x2, t) * density)

            self.get_v_average = get_v_average

            @njit
            def get_x1_average(density):
                return np.sum(x1 * density)

            self.get_x1_average = get_x1_average

            @njit
            def get_x2_average(density):
                return np.sum(x2 * density)

            self.get_x2_average = get_x2_average

            @njit
            def get_x1_average_rhs(density, t):
                return np.sum(diff_k_p1(p1, p2, t) * density)

            self.get_x1_average_rhs = get_x1_average_rhs

            @njit
            def get_x2_average_rhs(density, t):
                return np.sum(diff_k_p2(p1, p2, t) * density)

            self.get_x2_average_rhs = get_x2_average_rhs

            @njit
            def get_k_average(density, t):
                return np.sum(k(p1, p2, t) * density)

            self.get_k_average = get_k_average

            @njit
            def get_p1_average(density):
                return np.sum(p1 * density)

            self.get_p1_average = get_p1_average

            @njit
            def get_p2_average(density):
                return np.sum(p2 * density)

            self.get_p2_average = get_p2_average

            # Lists where the expectation values of x's and p's
            self.x1_average = []
            self.x2_average = []

            self.p1_average = []
            self.p2_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for x's and p's
            self.x1_average_rhs = []
            self.x2_average_rhs = []

            self.p1_average_rhs = []
            self.p2_average_rhs = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Allocate array for storing coordinate or momentum density of the wavefunction
            self.density = np.zeros(self.wavefunction.shape, dtype=np.float)

            # sequence of alternating signs for getting the wavefunction in the momentum representation
            self.minus = (-1) ** (k1 + k2)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.is_ehrenfest = True
        else:
            # Since diff_v and diff_k are not specified, we are not going to evaluate the Ehrenfest relations
            self.is_ehrenfest = False

    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """

        # pre-compute the sqrt of the volume element
        sqrt_dx1dx2 = np.sqrt(self.dx1 * self.dx2)

        for _ in range(time_steps):
            # make a half step in time
            self.t += 0.5 * self.dt

            # efficiently calculate
            #   wavefunction *= expV
            self.expV(self.wavefunction, self.t)

            # going to the momentum representation
            self.wavefunction = fftpack.fft2(self.wavefunction, overwrite_x=True)

            # efficiently evaluate
            #   wavefunction *= exp(-1j * dt * k)
            self.expK(self.wavefunction, self.t)

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft2(self.wavefunction, overwrite_x=True)

            # efficiently calculate
            #   wavefunction *= expV
            self.expV(self.wavefunction, self.t)

            # normalize
            # the following line is equivalent to
            # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2) * self.dX1 * self.dX2)
            # or
            self.wavefunction /= linalg.norm(self.wavefunction.reshape(-1)) * sqrt_dx1dx2

            # make a half step in time
            self.t += 0.5 * self.dt

            # calculate the Ehrenfest theorems
            self.get_ehrenfest()

        return self.wavefunction

    def get_ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems at time (t)
        """
        if self.is_ehrenfest:

            # evaluate the coordinate density
            np.abs(self.wavefunction, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current values of <x1> and <x2>
            self.x1_average.append(self.get_x1_average(self.density))
            self.x2_average.append(self.get_x2_average(self.density))

            self.p1_average_rhs.append(-self.get_p1_average_rhs(self.density, self.t))
            self.p2_average_rhs.append(-self.get_p2_average_rhs(self.density, self.t))

            # save the potential energy
            self.hamiltonian_average.append(self.get_v_average(self.density, self.t))

            # calculate density in the momentum representation
            wavefunction_p = fftpack.fft2(self.minus * self.wavefunction, overwrite_x=True)

            # get the density in the momentum space
            np.abs(wavefunction_p, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current values of <p1> and <p2>
            self.p1_average.append(self.get_p1_average(self.density))
            self.p2_average.append(self.get_p2_average(self.density))

            self.x1_average_rhs.append(self.get_x1_average_rhs(self.density, self.t))
            self.x2_average_rhs.append(self.get_x2_average_rhs(self.density, self.t))

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += self.get_k_average(self.density, self.t)

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 2D numpy array or a function specifying the wave function
        :return: self
        """
        if isinstance(wavefunc, (CPUDispatcher, FunctionType)):
            self.wavefunction[:] = wavefunc(self.x1, self.x2)

        elif isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape,\
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(np.complex))

        else:
            raise ValueError("wavefunc must be either string or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction.reshape(-1)) * np.sqrt(self.dx1 * self.dx2)

        return self