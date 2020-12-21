import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from numba import njit


class SplitOpSchrodinger1D(object):
    """
    The second-order split-operator propagator of the 1D Schrodinger equation
    in the coordinate representation
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    """
    def __init__(self, *, x_grid_dim, x_amplitude, v, k, dt, diff_k=None, diff_v=None, t=0, abs_boundary=1., **kwargs):
        """
        :param x_grid_dim: the grid size
        :param x_amplitude: the maximum value of the coordinates
        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)
        :param diff_k: the derivative of the potential energy for the Ehrenfest theorem calculations
        :param diff_v: the derivative of the kinetic energy for the Ehrenfest theorem calculations
        :param t: initial value of time
        :param dt: time increment
        :param abs_boundary: absorbing boundary
        :param kwargs: ignored
        """

        # saving the properties
        self.x_grid_dim = x_grid_dim
        self.x_amplitude = x_amplitude
        self.v = v
        self.k = k
        self.diff_v = diff_v
        self.diff_k = diff_k
        self.t = t
        self.dt = dt
        self.abs_boundary = abs_boundary

        # Check that all attributes were specified
        # make sure self.x_amplitude has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim, \
            "A value of the grid size (x_grid_dim) must be a power of 2"

        # get coordinate step size
        self.dx = 2. * self.x_amplitude / self.x_grid_dim

        # generate coordinate range
        x = self.x = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * self.dx
        # The same as
        # self.x = np.linspace(-self.x_amplitude, self.x_amplitude - self.dx , self.x_grid_dim)

        # generate momentum range as it corresponds to FFT frequencies
        p = self.p = (np.arange(self.x_grid_dim) - self.x_grid_dim / 2) * (np.pi / self.x_amplitude)

        # allocate the array for wavefunction
        self.wavefunction = np.zeros(self.x.size, dtype=np.complex)

        ####################################################################################################
        #
        # Codes for efficient evaluation
        #
        ####################################################################################################

        if isinstance(abs_boundary, (float, int, np.ndarray)):
            @njit
            def expV(wavefunction, t):
                """
                function to efficiently evaluate
                    wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
                """
                wavefunction *= (-1) ** np.arange(wavefunction.size) * abs_boundary * np.exp(-0.5j * dt * v(x, t))
        else:
            try:
                abs_boundary(x)
            except TypeError:
                raise ValueError("abs_boundary must be a numba function or a numerical constant or a numpy array")

            @njit
            def expV(wavefunction, t):
                """
                function to efficiently evaluate
                    wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
                """
                wavefunction *= (-1) ** np.arange(wavefunction.size) * abs_boundary(x) * np.exp(-0.5j * dt * v(x, t))

        self.expV = expV

        @njit
        def expK(wavefunction, t):
            """
            function to efficiently evaluate
                wavefunction *= exp(-1j * dt * k)
            """
            wavefunction *= np.exp(-1j * dt * k(p, t))

        self.expK = expK

        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        if diff_k and diff_v:

            # Get codes for efficiently calculating the Ehrenfest relations

            @njit
            def get_p_average_rhs(density, t):
                return np.sum(density * diff_v(x, t))

            self.get_p_average_rhs = get_p_average_rhs

            # The code above is equivalent to
            # self.get_p_average_rhs = njit(lambda density, t: np.sum(density * diff_v(x, t)))

            @njit
            def get_v_average(density, t):
                return np.sum(v(x, t) * density)

            self.get_v_average = get_v_average

            @njit
            def get_x_average(density):
                return np.sum(x * density)

            self.get_x_average = get_x_average

            @njit
            def get_x_average_rhs(density, t):
                return np.sum(diff_k(p, t) * density)

            self.get_x_average_rhs = get_x_average_rhs

            @njit
            def get_k_average(density, t):
                return np.sum(k(p, t) * density)

            self.get_k_average = get_k_average

            @njit
            def get_p_average(density):
                return np.sum(p * density)

            self.get_p_average = get_p_average

            # Lists where the expectation values of x and p
            self.x_average = []
            self.p_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for x and p
            self.x_average_rhs = []
            self.p_average_rhs = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # List where the expectation value of the kinetic energy will be stored
            self.k_average = []

            # List where the expectation value of the potential energy will be stored
            self.v_average = []

            # Allocate array for storing coordinate or momentum density of the wavefunction
            self.density = np.zeros(self.wavefunction.shape, dtype=np.float)

            # sequence of alternating signs for getting the wavefunction in the momentum representation
            self.minus = (-1) ** np.arange(self.x_grid_dim)

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
        for _ in range(time_steps):

            # advance the wavefunction by dt
            self.single_step_propagation()

            # calculate the Ehrenfest theorems
            self.get_ehrenfest()

        return self.wavefunction

    def single_step_propagation(self):
        """
        Perform a single step propagation of the wavefunction. The wavefunction is normalized.
        :return: self.wavefunction
        """
        # make a half step in time
        self.t += 0.5 * self.dt

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(self.wavefunction, self.t)

        # going to the momentum representation
        self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)

        # efficiently evaluate
        #   wavefunction *= exp(-1j * dt * k)
        self.expK(self.wavefunction, self.t)

        # going back to the coordinate representation
        self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)

        # efficiently evaluate
        #   wavefunction *= (-1) ** k * exp(-0.5j * dt * v)
        self.expV(self.wavefunction, self.t)

        # make a half step in time
        self.t += 0.5 * self.dt

        # normalize
        # this line is equivalent to
        # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction) ** 2 ) * self.dx)
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dx)

        return self.wavefunction

    def get_ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        if self.is_ehrenfest:
            # evaluate the coordinate density
            np.abs(self.wavefunction, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of <x>
            self.x_average.append(self.get_x_average(self.density))

            self.p_average_rhs.append(-self.get_p_average_rhs(self.density, self.t))

            # save the potential energy
            self.hamiltonian_average.append(self.get_v_average(self.density, self.t))

            # calculate density in the momentum representation
            wavefunction_p = fftpack.fft(self.minus * self.wavefunction, overwrite_x=True)

            # get the density in the momentum space
            np.abs(wavefunction_p, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of <p>
            self.p_average.append(self.get_p_average(self.density))

            self.x_average_rhs.append(self.get_x_average_rhs(self.density, self.t))

            # save the kinetic energy
            self.k_average.append(self.get_k_average(self.density, self.t))

            # save the potential energy
            self.v_average.append(self.get_v_average(self.density, 0))

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += self.k_average[-1]

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numpy array or function specifying the wave function
        :return: self
        """
        if isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape,\
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(np.complex))

        else:
            try:
                self.wavefunction[:] = wavefunc(self.x)
            except TypeError:
                raise ValueError("wavefunc must be either function or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dx)

        return self