from split_op_wigner_bloch import np, SplitOpWignerBloch
from numba import njit
import pyfftw


class CaldeiraLeggetMEq(SplitOpWignerBloch):
    """
    The class for propagating the Caldeira-Legget master equation using the methods from
        [*] https://arxiv.org/abs/1212.3406
    """
    def __init__(self, *, D=0, gamma=0, **kwargs):
        """
        :param D: the dephassing coefficient (see Eq. (36) in Ref. [*])
        :param gamma: the decay coefficient (see Eq. (62) in Ref. [*])
        :param kwargs: parameters passed to the parent class
        """
        # initialize the parent class
        SplitOpWignerBloch.__init__(self, **kwargs)

        # save new parameters
        self.D = D
        self.gamma = gamma

        # if the dephasing term is nonzero, modify the potential energy phase
        # Note that the dephasing term does not modify the Ehrenfest theorems
        if D:
            # introduce some aliases
            v = self.v
            x = self.x
            dt = self.dt
            Theta = self.Theta

            # Introduce the action of the dephasing term by redefining the potential term
            if self.time_independent_v:
                # pre-calculate the potential dependent phase since it is time-independent
                _expV = np.exp(-0.5j * dt * (v(x - 0.5 * Theta) - v(x + 0.5 * Theta)) - 0.5 * dt * D * Theta ** 2)

                @njit
                def expV(wignerfunction, t):
                    wignerfunction *= _expV

            else:
                # the phase factor is time-dependent
                @njit
                def expV(wignerfunction, t):
                    wignerfunction *= np.exp(-0.5j * dt * (
                                        v(x - 0.5 * Theta, t) - v(x + 0.5 * Theta, t) - 0.5 * dt * D * Theta ** 2
                                ))

            self.expV = expV

        # if the decay term is nonzero, prepare for
        if gamma:

            # allocate an array for the extra copy of the Wigner function, i.e., for W^{(1)} in Eq. (63) of Ref. [*]
            self.wigner_1 = pyfftw.empty_aligned(self.wignerfunction.shape, dtype=self.wignerfunction.dtype)

            # p x -> theta x for self.wigner_1
            self.wigner_1_transform_p2theta = pyfftw.builders.rfft(self.wigner_1, axis=0, **self.fft_params)

            # theta x  ->  p x for self.wigner_1
            self.wigner_1_transform_theta2p = pyfftw.builders.irfft(
                self.wigner_1_transform_p2theta(), axis=0, **self.fft_params
            )

    def single_step_propagation(self):
        """
        Overload the method in the parent class.
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """

        # In order to get the third order propagator, we incorporate the decay term (Eq. (62) from Ref. [*]) using
        # the splitting scheme where we first apply the decay term for dt / 2, then the unitary propagator for full dt,
        # and finally the decay term for dt / 2.

        self.decay_term_half_step()
        SplitOpWignerBloch.single_step_propagation(self)
        self.decay_term_half_step()

        return self.wignerfunction

    def decay_term_half_step(self):
        """
        Apply the decay term (Eq. (62) from Ref. [*]) for half a time-step.
        :return: None
        """
        if self.gamma:

            # declare aliases
            wigner_1 = self.wigner_1

            np.copyto(wigner_1, self.wignerfunction)

            # Eq. (65) of Ref. [*]
            wigner_1 *= self.p
            wigner_1 = self.wigner_1_transform_p2theta(wigner_1)

            wigner_1 *= self.Theta
            wigner_1 *= 0.5j * self.dt * self.gamma
            wigner_1 = self.wigner_1_transform_theta2p(wigner_1)

            # Eq. (64) of Ref. [*]
            wigner_1 += self.wignerfunction

            # Eq. (63) of Ref. [*]
            #   it starts with doing Eq. (65) again
            wigner_1 *= self.p
            wigner_1 = self.wigner_1_transform_p2theta(wigner_1)
            wigner_1 *= self.Theta
            wigner_1 *= 1j * self.dt * self.gamma
            wigner_1 = self.wigner_1_transform_theta2p(wigner_1)

            self.wignerfunction += wigner_1

            self.wigner_1 = wigner_1