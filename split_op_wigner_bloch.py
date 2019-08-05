from split_op_wigner_moyal import SplitOpWignerMoyal, np
from functools import partial
import warnings


class SplitOpWignerBloch(SplitOpWignerMoyal):
    """
    The second-order split-operator propagator for
    finding the Wigner function of the Maxwell-Gibbs canonical state [rho = exp(-beta * H)]
    by split-operator propagation of the Bloch equation in phase space.

    Details about the method can be found at https://arxiv.org/abs/1602.07288

    The Hamiltonian should be of the form H = k(p) + v(x).

    This implementation follows split_op_wigner_moyal.py
    """
    def __init__(self, *, dbeta=None, beta=None, **kwargs):
        """
        :param dbeta: inverse temperature time step
        :param beta: inverse temperature
        :param kwargs: the rest of the arguments to be passed to the parent class
        """
        SplitOpWignerMoyal.__init__(self, **kwargs)

        self.dbeta = dbeta
        self.beta = beta

    def setup_bloch_propagator(self):
        """
        Pre-calculate exponents used for the split operator propagation
        """
        v = (self.v if self.time_independent_v else partial(self.v, t=self.t))
        k = (self.k if self.time_independent_k else partial(self.k, t=self.t))

        # Get the sum of the potential energy contributions
        self.bloch_expV = -0.25 * self.dbeta * (
            v(self.x - 0.5 * self.Theta) + v(self.x + 0.5 * self.Theta)
        )

        # Make sure that the largest value is zero
        self.bloch_expV -= self.bloch_expV.max()

        # such that the following exponent is never larger then one
        np.exp(self.bloch_expV, out=self.bloch_expV)

        # Get the sum of the kinetic energy contributions
        self.bloch_expK = -0.5 * self.dbeta *(
                k(self.p + 0.5 * self.Lambda) + k(self.p - 0.5 * self.Lambda)
        )

        # Make sure that the largest value is zero
        self.bloch_expK -= self.bloch_expK.max()

        # such that the following exponent is never larger then one
        np.exp(self.bloch_expK, out=self.bloch_expK)

        # Initialize the Wigner function as the infinite temperature Gibbs state
        self.set_wignerfunction(lambda x, p: 1. + 0. * x + 0. * p)

    def single_step_bloch_propagation(self):
        """
        Advance thermal state calculation by self.dbeta using the second order Bloch propagator
        :return:
        """
        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)
        self.wignerfunction *= self.bloch_expV

        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        # p x  ->  p lambda
        self.wignerfunction = self.transform_x2lambda(self.wignerfunction)
        self.wignerfunction *= self.bloch_expK

        # p lambda  ->  p x
        self.wignerfunction = self.transform_lambda2x(self.wignerfunction)

        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)
        self.wignerfunction *= self.bloch_expV

        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        # normalization
        self.wignerfunction /= self.wignerfunction.sum() * self.dxdp

    def get_thermal_state(self, beta=None, nsteps=5000, max_purity=0.9999):
        """
        Calculate the thermal state via the Bloch propagator
        :param beta: inverse temperature (default beta = self.beta)
        :param max_purity: maximum value of purity to be allowed
        :return: self.wignerfunction containing the thermal state
        """
        # get the inverse temperature increment
        beta = (beta if beta else self.beta)
        self.dbeta = beta / nsteps

        self.setup_bloch_propagator()

        for _ in range(nsteps):
            self.single_step_bloch_propagation()

            # check that the purity of the state does not exceed one
            if self.get_purity(self.wignerfunction) > max_purity:
                warnings.warn("purity reached the maximum")
                break

        return self.wignerfunction

    def get_ground_state(self, dbeta=None, max_purity=0.9999):
        """
        Calculate the Wigner function of the ground state as a zero temperature Gibbs state
        :param max_purity: maximum value of purity to be allowed
        :return: self.wignerfunction
        """
        self.dbeta = (dbeta if dbeta else 2. * self.dt)

        self.setup_bloch_propagator()

        while self.get_purity(self.wignerfunction) < max_purity:
            self.single_step_bloch_propagation()

        return self.wignerfunction
