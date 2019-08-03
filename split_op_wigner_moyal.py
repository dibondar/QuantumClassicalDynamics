import numpy as np
# in other codes, we have used scipy.fftpack to perform Fourier Transforms.
# In this code, we will use pyfftw, which is more suited for efficient large data
import pyfftw
from numba import njit
from numba.targets.registry import CPUDispatcher
from types import FunctionType
from multiprocessing import cpu_count


class SplitOpWignerMoyal(object):
    """
    The second-order split-operator propagator for the Moyal equation for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    (K and V may not depend on time.)

    This implementation using PyFFTW.

    Details about the method can be found at https://arxiv.org/abs/1212.3406

    This implementation stores the Wigner function as a 2D real array.
    """
    def __init__(self, *, x_grid_dim, x_amplitude, p_grid_dim, p_amplitude, dt, k, v, t=0,
                 diff_k=None, diff_v=None, time_independent_v=True, time_independent_k=True, threads=-1, **kwargs):
        """
        :param x_grid_dim: the coordinate grid size
        :param p_grid_dim: the momentum grid size

        :param x_amplitude: the maximum value of the coordinates
        :param p_amplitude: the maximum value of the momentum

        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)

        :param diff_k: the derivative of the potential energy for the Ehrenfest theorem calculations
        :param diff_v: the derivative of the kinetic energy for the Ehrenfest theorem calculations

        :param time_independent_v: boolean flag indicated weather potential is time dependent (default is True)
        :param time_independent_k: boolean flag indicated weather kinetic energy is time dependent (default is True)

        :param t: initial value of time
        :param dt: initial time increment

        :param threads: number of threads to be used for FFT (default all)

        :param kwargs: ignored
        """

        # save all attributes
        self.x_grid_dim = x_grid_dim
        self.p_grid_dim = p_grid_dim
        self.x_amplitude = x_amplitude
        self.p_amplitude = p_amplitude
        self.v = v
        self.time_independent_v = time_independent_v
        self.k = k
        self.time_independent_k = time_independent_k
        self.diff_v = diff_v
        self.t = t
        self.dt = dt

        # make sure self.x_grid_dim and self.p_grid_dim has a value of power of 2
        assert 2 ** int(np.log2(self.p_grid_dim)) == self.x_grid_dim and \
               2 ** int(np.log2(self.p_grid_dim)) == self.p_grid_dim, \
            "A value of the grid sizes (x_grid_dim and p_grid_dim) must be a power of 2"

        ########################################################################################
        #
        #   Initialize Fourier transform for efficient calculations
        #
        ########################################################################################

        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()

        # allocate the array for Wigner function
        self.wignerfunction = pyfftw.empty_aligned((self.p_grid_dim, self.x_grid_dim), dtype=np.float)

        threads = (cpu_count() if threads < 1 else threads)

        # p x -> theta x
        self.transform_p2theta = pyfftw.builders.rfft(
            self.wignerfunction, axis=0,
            overwrite_input=True,  avoid_copy=True, threads=threads,
        )

        # theta x  ->  p x
        self.transform_theta2p = pyfftw.builders.irfft(
            self.transform_p2theta(), axis=0,
            overwrite_input=True, avoid_copy=True, threads=threads,
        )

        # p x  ->  p lambda
        self.transform_x2lambda = pyfftw.builders.rfft(
            self.wignerfunction, axis=1,
            overwrite_input=True, avoid_copy=True, threads=threads,
        )

        # p lambda  ->  p x
        self.transform_lambda2x = pyfftw.builders.irfft(
            self.transform_x2lambda(), axis=1,
            overwrite_input=True, avoid_copy=True, threads=threads,
        )

        ########################################################################################
        #
        #   Initialize grids
        #
        ########################################################################################

        # get coordinate and momentum step sizes
        self.dx = 2. * self.x_amplitude / self.x_grid_dim
        self.dp = 2. * self.p_amplitude / self.p_grid_dim

        # pre-compute the volume element in phase space
        dxdp = self.dxdp = self.dx * self.dp

        # generate coordinate and momentum ranges
        # see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # for explanation of np.newaxis and other array indexing operations
        # also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # for understanding the broadcasting in array operations

        x = self.x = (np.arange(self.x_grid_dim)[np.newaxis, :] - self.x_grid_dim / 2) * self.dx
        p = self.p = (np.arange(self.p_grid_dim)[:, np.newaxis] - self.p_grid_dim / 2) * self.dp

        # Lambda grid (variable conjugate to the coordinate)
        # (take only first half, as required by the real fft)
        Lambda = self.Lambda = np.arange(1 + self.x_grid_dim // 2)[np.newaxis, :] * (np.pi / self.x_amplitude)

        # Theta grid (variable conjugate to the momentum)
        # (take only first half, as required by the real fft)
        Theta = self.Theta = np.arange(1 + self.p_grid_dim // 2)[:, np.newaxis] * (np.pi / self.p_amplitude)

        # Define the functions performing multiplication by the phase factor associated with the potential energy
        if time_independent_v:
            # pre-calculate the potential dependent phase since it is time-independent
            _expV = np.exp(-0.5j * dt * (v(x - 0.5 * Theta) - v(x + 0.5 * Theta)))

            @njit
            def expV(wignerfunction, t):
                wignerfunction *= _expV

        else:
            # the phase factor is time-dependent
            @njit
            def expV(wignerfunction, t):
                wignerfunction *= np.exp(-0.5j * dt * (
                                    v(x - 0.5 * Theta, t) - v(x + 0.5 * Theta, t)
                            ))

        self.expV = expV

        # Define the functions performing multiplication by the phase factor associated with the kinetic energy
        if time_independent_k:
            # pre-calculate the kinetic energy dependent phase since it is time-independent
            _expK = np.exp(-dt * 1j * (k(p + 0.5 * Lambda) - k(p - 0.5 * Lambda)))

            @njit
            def expK(wignerfunction, t):
                wignerfunction *= _expK

        else:
            # the phase factor is time-dependent
            @njit
            def expK(wignerfunction, t):
                wignerfunction *= np.exp(-dt * 1j * (
                                k(p + 0.5 * Lambda, t) - k(p - 0.5 * Lambda, t)
                        ))

        self.expK = expK

        # method for calculating the purity of the wigner function
        self.get_purity = njit(
            lambda wignerfunction: 2. * np.pi * np.sum(wignerfunction ** 2) * dxdp
        )

        # Check whether the necessary terms are specified to calculate the Ehrenfest theorems
        if diff_v and diff_k:

            # codes to calculate the first-order Ehrenfest theorems
            self.get_x_average = njit(lambda wignerfunction: np.sum(wignerfunction * x))
            self.get_p_average = njit(lambda wignerfunction: np.sum(wignerfunction * p))

            if time_independent_v:
                _diff_v = diff_v(x)
                self.get_p_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * _diff_v))

                _v = v(x)
                self.get_v_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * _v))
            else:
                self.get_p_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * diff_v(x, t)))
                self.get_v_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * v(x, t)))

            if time_independent_k:
                _diff_k = diff_k(p)
                self.get_x_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * _diff_k))

                _k = k(p)
                self.get_k_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * _k))
            else:
                self.get_x_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * diff_k(p, t)))
                self.get_k_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * k(p, t)))

            # since the variable time propagator is used, we record the time when expectation values are calculated
            self.times = []

            # Lists where the right hand sides of the Ehrenfest theorems for x and p
            self.x_average_rhs = []
            self.p_average_rhs = []

            # Lists where the expectation values of x and p
            self.x_average = []
            self.p_average = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        else:
            # Since self.diff_v and self.diff_k are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

    def propagate(self, time_steps=1):
        """
        Time propagate the Wigner function saved in self.wignerfunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wignerfunction
        """
        for _ in range(time_steps):
            # advance by one time step
            self.single_step_propagation()

            # normalization
            self.wignerfunction /= self.wignerfunction.sum() * self.dxdp

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest()

        return self.wignerfunction

    def single_step_propagation(self):
        """
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        # make a half step in time
        self.t += 0.5 * self.dt

        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)

        self.expV(self.wignerfunction, self.t)

        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        # p x  ->  p lambda
        self.wignerfunction = self.transform_x2lambda(self.wignerfunction)
        self.expK(self.wignerfunction, self.t)

        # p lambda  ->  p x
        self.wignerfunction = self.transform_lambda2x(self.wignerfunction)

        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)
        self.expV(self.wignerfunction, self.t)

        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        # make a half step in time
        self.t += 0.5 * self.dt

        return self.wignerfunction

    def get_Ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems at time (t)
        """
        if self.isEhrenfest:
            wignerfunction = self.wignerfunction
            dxdp = self.dxdp
            t = self.t

            self.x_average.append(self.get_x_average(wignerfunction) * dxdp)
            self.p_average.append(self.get_p_average(wignerfunction) * dxdp)

            self.p_average_rhs.append(-self.get_p_average_rhs(wignerfunction, t) * dxdp)

            self.x_average_rhs.append(self.get_x_average_rhs(wignerfunction, t) * dxdp)

            self.hamiltonian_average.append(
                self.get_k_average(wignerfunction, t) * dxdp
                +
                self.get_v_average(wignerfunction, t) * dxdp
            )

            # save the current time
            self.times.append(t)

    def set_wignerfunction(self, new_wignerfunction):
        """
        Set the initial Wigner function
        :param new_wignerfunction: a 2D numpy array or sting containing the wigner function
        :return: self
        """
        if isinstance(new_wignerfunction, (CPUDispatcher, FunctionType)):

            self.wignerfunction[:] = new_wignerfunction(self.x, self.p)

        elif isinstance(new_wignerfunction, np.ndarray):

            assert new_wignerfunction.shape == self.wignerfunction.shape, \
                "The grid sizes does not match with the Wigner function"

            # save only real part
            np.copyto(self.wignerfunction, new_wignerfunction.real)

        else:
            raise ValueError("Wigner function must be either a function or numpy.array")

        # normalize
        self.wignerfunction /= self.wignerfunction.sum() * self.dxdp

        return self