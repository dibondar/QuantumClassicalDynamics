import numpy as np
# in other codes, we have used scipy.fftpack to perform Fourier Transforms.
# In this code, we will use pyfftw, which is more suited for efficient large data
import pyfftw
import pickle
from numba import njit
from numba.core.registry import CPUDispatcher
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
    def __init__(self, *, x_grid_dim, x_amplitude, p_grid_dim, p_amplitude, dt, k, v, t=0, D=0, p_rhs=None, x_rhs=None,
                 threads=-1, fftw_wisdom_fname='fftw.wisdom', planner_effort='FFTW_MEASURE', **kwargs):
        """
        The Wigner function propagator of the Moyal equation of motion.
        The Hamiltonian should be of the form H = k(p) + v(x).
        The potential v(x) and kinetic k(p) energies may depend on time. In that case,
        set time_independent_v = False and time_independent_k = False.

        :param x_grid_dim: the coordinate grid size
        :param p_grid_dim: the momentum grid size

        :param x_amplitude: the maximum value of the coordinates
        :param p_amplitude: the maximum value of the momentum

        :param v: the potential energy (as a function)
        :param k: the kinetic energy (as a function)

        :param x_rhs: the rhs of the Ehrenfest theorem for x
        :param p_rhs: the rhs of the Ehrenfest theorem for p

        :param t: initial value of time
        :param dt: initial time increment

        :param D: the decoherence term (i.e., quantum diffusion). see Eq. (36) of https://arxiv.org/abs/1212.3406

        :param threads: number of threads to be used for FFT (default all)

        :param fftw_wisdom_fname: File name from where the FFT wisdom will be loaded from and saved to

        :param kwargs: ignored
        :param planner_effort: a string dictating how much effort is spent in planning the FFTW routines
        """

        # save all attributes
        self.x_grid_dim = x_grid_dim
        self.p_grid_dim = p_grid_dim
        self.x_amplitude = x_amplitude
        self.p_amplitude = p_amplitude
        self.v = v
        self.k = k
        self.p_rhs = p_rhs
        self.x_rhs = x_rhs
        self.t = t
        self.dt = dt
        self.D = D

        # make sure self.x_grid_dim and self.p_grid_dim has a value of power of 2
        assert 2 ** int(np.log2(self.x_grid_dim)) == self.x_grid_dim and \
               2 ** int(np.log2(self.p_grid_dim)) == self.p_grid_dim, \
            "A value of the grid sizes (x_grid_dim and p_grid_dim) must be a power of 2"

        ########################################################################################
        #
        #   Initialize Fourier transform for efficient calculations
        #
        ########################################################################################

        # Load the FFTW wisdom
        try:
            with open(fftw_wisdom_fname, 'rb') as fftw_wisdow:
                pyfftw.import_wisdom(pickle.load(fftw_wisdow))
        except FileNotFoundError:
            pass

        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()

        # allocate the array for Wigner function
        self.wignerfunction = pyfftw.empty_aligned((self.p_grid_dim, self.x_grid_dim), dtype=np.float)

        threads = (cpu_count() if threads < 1 else threads)

        # parameters for FFT
        self.fft_params = {
            "overwrite_input": True,
            "avoid_copy": True,
            "threads": threads,
            "planner_effort": planner_effort,
        }

        # p x -> theta x
        self.transform_p2theta = pyfftw.builders.rfft(self.wignerfunction, axis=0, **self.fft_params)

        # theta x  ->  p x
        self.transform_theta2p = pyfftw.builders.irfft(self.transform_p2theta(), axis=0, **self.fft_params)

        # p x  ->  p lambda
        self.transform_x2lambda = pyfftw.builders.rfft(self.wignerfunction, axis=1, **self.fft_params)

        # p lambda  ->  p x
        self.transform_lambda2x = pyfftw.builders.irfft(self.transform_x2lambda(), axis=1, **self.fft_params)

        # Save the FFTW wisdom
        with open(fftw_wisdom_fname, 'wb') as fftw_wisdow:
            pickle.dump(pyfftw.export_wisdom(), fftw_wisdow)

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

        # Decide whether the potential depends on time
        try:
            v(x, 0)
            time_independent_v = False
        except TypeError:
            time_independent_v = True
        self.time_independent_v = time_independent_v

        # Decide whether the kinetic energy depends on time
        try:
            k(p, 0)
            time_independent_k = False
        except TypeError:
            time_independent_k = True
        self.time_independent_k = time_independent_k

        # Define the functions performing multiplication by the phase factor associated with the potential energy
        if time_independent_v:
            # pre-calculate the potential dependent phase since it is time-independent
            _expV = np.exp(
                -0.5j * dt * (v(x - 0.5 * Theta) - v(x + 0.5 * Theta))
                -0.5 * dt * D * Theta ** 2 # the decoherence term
            )

            @njit
            def expV(wignerfunction, t):
                wignerfunction *= _expV

        else:
            v = njit(v)

            # the phase factor is time-dependent
            @njit
            def expV(wignerfunction, t):
                wignerfunction *= np.exp(-0.5j * dt * (
                                    v(x - 0.5 * Theta, t) - v(x + 0.5 * Theta, t)
                                    )
                                    -0.5 * dt * D * Theta ** 2 # the decoherence term
                                )

        self.expV = expV

        # Define the functions performing multiplication by the phase factor associated with the kinetic energy
        if time_independent_k:
            # pre-calculate the kinetic energy dependent phase since it is time-independent
            _expK = np.exp(-dt * 1j * (k(p + 0.5 * Lambda) - k(p - 0.5 * Lambda)))

            @njit
            def expK(wignerfunction, t):
                wignerfunction *= _expK

        else:
            k = njit(k)

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
        if p_rhs and x_rhs:

            # codes to calculate the first-order Ehrenfest theorems
            self.get_x_average = njit(lambda wignerfunction: np.sum(wignerfunction * x) * dxdp)
            self.get_p_average = njit(lambda wignerfunction: np.sum(wignerfunction * p) * dxdp)

            if time_independent_v:
                _p_rhs = p_rhs(x, p)
                self.get_p_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * _p_rhs) * dxdp)

                _v = v(x)
                self.get_v_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * _v * dxdp))
            else:
                p_rhs = njit(p_rhs)

                self.get_p_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * p_rhs(x, p, t)) * dxdp)
                self.get_v_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * v(x, t)) * dxdp)

            if time_independent_k:
                _x_rhs = x_rhs(p)
                self.get_x_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * _x_rhs) * dxdp)

                _k = k(p)
                self.get_k_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * _k) * dxdp)
            else:
                x_rhs = njit(x_rhs)

                self.get_x_average_rhs = njit(lambda wignerfunction, t: np.sum(wignerfunction * x_rhs(x, p, t)) * dxdp)
                self.get_k_average = njit(lambda wignerfunction, t: np.sum(wignerfunction * k(p, t)) * dxdp)

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

            self.x_average.append(self.get_x_average(wignerfunction))
            self.p_average.append(self.get_p_average(wignerfunction))

            self.p_average_rhs.append(self.get_p_average_rhs(wignerfunction, t))

            self.x_average_rhs.append(self.get_x_average_rhs(wignerfunction, t))

            self.hamiltonian_average.append(
                self.get_k_average(wignerfunction, t) + self.get_v_average(wignerfunction, t)
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