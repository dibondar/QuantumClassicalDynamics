from split_op_schrodinger1D import SplitOpSchrodinger1D, fftpack, np, linalg

# We will use the inheritance in the object orienting programing (see, e.g.,
# https://en.wikipedia.org/wiki/Inheritance_%28object-oriented_programming%29 and
# https://docs.python.org/2/tutorial/classes.html)
# to add methods to already developed propagator (SplitOpSchrodinger1D)
# that find stationary states via the imaginary-time propagation


class ImgTimePropagation(SplitOpSchrodinger1D):

    def get_stationary_states(self, nstates, nsteps=10000):
        """
        Obtain stationary states via the imaginary time propagation
        :param nstates: number of states to obtaine.
                If nstates = 1, only the ground state is obtained. If nstates = 2,
                the ground and first exited states are obtained, etc
        :param nsteps: number of the imaginary time steps to take
        :return:self
        """
        # since there is no time dependence (self.t) during the imaginary time propagation
        # pre-calculate imaginary time exponents of the potential and kinetic energy
        # img_expV = ne.evaluate("(-1) ** k * exp(-0.5 * dt * ({}))".format(self.V), local_dict=vars(self))

        img_exp_v = (-1) ** np.arange(self.wavefunction.size) * np.exp(-0.5 * self.dt * self.v(self.x, self.t))

        # img_expK = ne.evaluate("exp(-dt * ({}))".format(self.K), local_dict=vars(self))
        img_exp_k = np.exp(-self.dt * self.k(self.p, self.t))


        # initialize the list where the stationary states will be saved
        self.stationary_states = []

        # boolean flag determining the parity of wavefunction
        even = True

        for n in range(nstates):

            # allocate and initialize the wavefunction depending on the parity.
            # Note that you do not have to be fancy and can choose any initial guess (even random).
            # the more reasonable initial guess, the faster the convergence.
            wavefunction = (np.exp(-self.x ** 2) if even else self.x * np.exp(-self.x ** 2))
            even = not even

            for _ in range(nsteps):
                #################################################################################
                #
                #   Make an imaginary time step
                #
                #################################################################################
                wavefunction *= img_exp_v

                # going to the momentum representation
                wavefunction = fftpack.fft(wavefunction, overwrite_x=True)
                wavefunction *= img_exp_k

                # going back to the coordinate representation
                wavefunction = fftpack.ifft(wavefunction, overwrite_x=True)
                wavefunction *= img_exp_v

                #################################################################################
                #
                #    Project out all previously calculated stationary states
                #
                #################################################################################

                # normalize
                wavefunction /= linalg.norm(wavefunction) * np.sqrt(self.dx)

                # calculate the projections
                projs = [np.vdot(psi, wavefunction) * self.dx for psi in self.stationary_states]

                # project out the stationary states
                for psi, proj in zip(self.stationary_states, projs):
                    wavefunction -= proj * psi
                    # ne.evaluate("wavefunction - proj * psi", out=wavefunction)

                # normalize
                wavefunction /= linalg.norm(wavefunction) * np.sqrt(self.dx)

            # save obtained approximation to the stationary state
            self.stationary_states.append(wavefunction)

        return self
