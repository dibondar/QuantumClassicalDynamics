import numpy as np
from split_op_wigner_moyal import SplitOpWignerMoyal


class CaldeiraLeggetMEq(SplitOpWignerMoyal):
    """
    The class for propagating the Caldeira-Legget master equation using the methods from
        [*] https://arxiv.org/abs/1212.3406
    """
    def __init__(self, *, gamma=0, **kwargs):
        """
        Propagator for the Caldeira-Leggett model
        :param gamma: the decay coefficient in the dissipatr D[rho] = -gamma * [x, {p, rho}] (See Eq. (62) of [*]).
        :param kwargs: parameters to be passed to the parent class SplitOpWignerMoyal
        """
        self.gamma = gamma
        super().__init__(**kwargs)

        # an array for storing the copy of the wigner function is needed for method self.apply_friction_half_dt
        self.wignerfunction_copy = np.empty_like(self.wignerfunction)

    def single_step_propagation(self):
        """
        Overload the method of the parent class.
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        # In order to get the third order propagator, we incorporate the decay term (Eq. (62) from Ref. [*]) using
        # the splitting scheme where we first apply the decay term for dt / 2, then the unitary propagator for full dt,
        # and finally the decay term for dt / 2.
        if self.gamma:
            self.apply_friction_half_dt()

        super(CaldeiraLeggetMEq, self).single_step_propagation()

        if self.gamma:
            self.apply_friction_half_dt()

    def apply_friction_half_dt(self):
        """
        Applying the friction dissipator onto the wigner function for half time step
        utilizing Eq. (62) of https://arxiv.org/abs/1212.3406
        :return: None
        """
        # copy the current wigner function
        np.copyto(self.wignerfunction_copy, self.wignerfunction)

        ##############################################################
        #
        # Eq. (65) of Ref. [*]
        #
        ##############################################################
        self.wignerfunction *= self.p
        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)
        self.wignerfunction *= self.Theta
        self.wignerfunction *= 0.5j * self.dt * self.gamma
        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        ##############################################################
        #
        # Eq. (64) of Ref. [*]
        #
        ##############################################################
        self.wignerfunction += self.wignerfunction_copy

        ##############################################################
        #
        # Eq. (63) of Ref. [*]
        # starts with doing Eq. (65) again
        #
        ##############################################################
        self.wignerfunction *= self.p
        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)
        self.wignerfunction *= self.Theta
        self.wignerfunction *= 1j * self.dt * self.gamma
        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        ##############################################################
        #
        # Eq. (63) of Ref. [*]
        #
        ##############################################################
        self.wignerfunction += self.wignerfunction_copy