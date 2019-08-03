import numpy as np
from types import MethodType, FunctionType
import warnings


class VerletIntegrator:
    """
    The Verlet integrator of classical dynamics, which is the second-order symplectic integrator.
    Note that the Verlet integrator applies to autonomous systems, whose Hamiltonian is time independent.

    This implementation is valid for any spatial dimension.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified:

            V(x0, x1, ...) - (optional) time-independent potential energy
            grad_V(x0, x1, ...) - potential energy gradient (returns [diff(V, p0), diff(V, p1), ...])

            K(p0, p1, ...) - (optional) time-independent kinetic energy
            grad_K(p0, p1, ...) - kinetic energy gradient (returns [diff(K, p0), diff(K, p1), ...])
        """
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            # otherwise bind it as a property
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.grad_V
        except AttributeError:
            raise AttributeError("Potential energy gradient (grad_V) was not specified")

        try:
            self.grad_K
        except AttributeError:
            raise AttributeError("Kinetic energy gradient (grad_K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            warnings.warn("initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        try:
            # check whether kinetic and potential energy
            self.V
            self.K

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        except AttributeError:
            # Since self.V and self.K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

    def set_ensemble(self, X, P, weights=None):
        """
        Set initial condition for classical ensemble
        :param X: array-like object of dimensions (number of spatial variables, number classical particles)
                specifying initial positions of the classical particles in the ensemble.

        :param P: array-like object of dimensions (number of spatial variables, number classical particles)
                specifying initial momenta of the classical particles in the ensemble.

        :param weights: (optional) weights of classical particles. Weights are assumed to be normilized.
            If not specified, the equally distributed weights are assigned.

        :return: self
        """
        self.X = np.array(X)
        self.P = np.array(P)

        # consistency checks
        assert len(self.X.shape) == 2
        assert self.X.shape == self.P.shape, "Coordinate (X) and momenta (P) must have the same size"

        # extract number of classical particles in the ensemble
        n_particles = self.X.shape[1]

        # If not specified, the equally distributed weights are assigned
        self.weights = (np.ones(n_particles) / n_particles if weights == None else np.array(weights))

        assert self.weights.shape == (n_particles,), "Number of weights must be consistent with specified X and P"

        return self

    def propagate(self, time_steps=1):
        """
        Time propagate the classical ensemble
        :param time_steps: number of self.dt time increments to make
        :return: (self.X, self.P) updated positions and momenta of classical particles
        """
        for _ in range(time_steps):

            # Perfrom the Verlet propagation
            # Step 1 (update momentum): P1 = P - grad_V(X) * dt/2
            tmp = np.array(self.grad_V(*self.X), copy=False)
            tmp *= self.dt/2
            self.P -= tmp

            # Step 2 (update coordinate): X1 = X + grad_K(P1) * dt
            tmp = np.array(self.grad_K(*self.P), copy=False)
            tmp *= self.dt
            self.X += tmp

            # Step 3 (update momentum): P2 = P1 - grad_V(X1) * dt/2
            tmp = np.array(self.grad_V(*self.X), copy=False)
            tmp *= self.dt/2
            self.P -= tmp

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest()

            # increment current time
            self.t += self.dt

        return self.X, self.P

    def get_Ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems.
        """
        if self.isEhrenfest:

            # Verify the first Ehrenfest theorem
            self.X_average.append(
                tuple(np.dot(self.weights, _) for _ in self.X)
            )

            self.X_average_RHS.append(
                tuple(np.dot(self.weights, _) for _ in self.grad_K(*self.P))
            )

            # Verify the second Ehrenfest theorem
            self.P_average.append(
                tuple(np.dot(self.weights, _) for _ in self.P)
            )

            self.P_average_RHS.append(
                tuple(-np.dot(self.weights, _) for _ in self.grad_V(*self.X))
            )

            # save expectation value of Hamiltonian
            self.hamiltonian_average.append(
                np.dot(self.weights, self.K(*self.P) + self.V(*self.X))
            )

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # Plotting facility
    import matplotlib.pyplot as plt

    # Use the documentation string for the developed class
    print(VerletIntegrator.__doc__)

    ##############################################################################
    #
    #   1D Test
    #
    ##############################################################################

    #plt.subplot(111, axisbg='grey')
    plt.subplot(111)
    plt.title('1D Dynamical Billiard (phase space)')

    # Initiate classical system
    sys = VerletIntegrator(
        V=lambda self, x: 10 * (1. - np.exp(-0.2 * x ** 2)),
        grad_V=lambda self, x: (4 * x * np.exp(-0.2 * x ** 2), ),

        K=lambda self, p: 0.5 * p ** 2,
        grad_K=lambda self, p: (p, ),

        dt=0.001
    )

    # Set initial conditions
    sys.set_ensemble(
        X=np.random.uniform(-1.5, 1.5, (1, 10)),
        P=np.random.uniform(-1.5, 1.5, (1, 10))
    )

    # propagate
    c = np.arange(sys.X.shape[1], dtype=np.float).reshape(sys.X.shape)
    for _ in range(2000):
        X, P = sys.propagate(100)
        plt.scatter(X, P, c=c, s=1, edgecolors='face')

    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    var_hamiltonian = 100. * (max(sys.hamiltonian_average) / min(sys.hamiltonian_average) - 1.)
    print("\nHamilton variation {:.2e} percent\n".format(var_hamiltonian))

    plt.show()

    ##############################################################################

    plt.subplot(121)
    plt.title("Verify the first Ehrenfest theorem")

    times = sys.dt * np.arange(len(sys.X_average))
    plt.plot(
        times,
        np.gradient(np.ravel(sys.X_average), sys.dt),
        '-r',
        label='$d\\langle x \\rangle / dt$'
    )
    plt.plot(
        times,
        np.ravel(sys.X_average_RHS),
        '--b',
        label='$\\langle p \\rangle$'
    )
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(122)
    plt.title("Verify the second Ehrenfest theorem")

    plt.plot(
        times,
        np.gradient(np.ravel(sys.P_average), sys.dt),
        '-r',
        label='$d\\langle p \\rangle / dt$'
    )
    plt.plot(
        times,
        np.ravel(sys.P_average_RHS),
        '--b',
        label='$\\langle -U\'(x)\\rangle$'
    )
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.show()

    ##############################################################################
    #
    #   3D Test
    #
    ##############################################################################

    #plt.subplot(111, axisbg='grey')
    plt.subplot(111)
    plt.title('3D Dynamical Billiard (phase-space projection)')

    def grad_V(self, x1, x2, x3):
        tmp = 10 * np.exp(-x1 ** 2 - 0.2 * x2 ** 2 - 0.1 * x3 ** 2)
        return 2 * x1 * tmp, 0.4 * x2 * tmp,  0.2 * x3 * tmp

    # Initiate classical system
    sys = VerletIntegrator(
        V=lambda self, x1, x2, x3: 10 * (1. - np.exp(-x1 ** 2 -0.2 * x2 ** 2 -0.1 * x3 ** 2)),
        grad_V=grad_V, # defined above

        K=lambda self, p1, p2, p3: 0.5 * (p1 ** 2 + p2 ** 2 + p3 ** 2),
        grad_K=lambda self, p1, p2, p3: (p1, p2, p3),

        dt=0.001
    )

    # Set initial conditions
    sys.set_ensemble(
        X=np.random.uniform(-1., 1., (3, 10)),
        P=np.random.uniform(-1., 1., (3, 10))
    )

    # propagate
    c = np.arange(sys.X.shape[1])
    for _ in range(2000):
        X, P = sys.propagate(150)
        plt.scatter(X[0], P[0], s=2, c=c, edgecolors='face')

    plt.xlabel('$x_1$ (a.u.)')
    plt.ylabel('$p_1$ (a.u.)')

    var_hamiltonian = 100. * (max(sys.hamiltonian_average) / min(sys.hamiltonian_average) - 1.)
    print("\nHamilton variation {:.2e} percent\n".format(var_hamiltonian))

    plt.show()

    ##############################################################################

    plt.subplot(121)
    plt.title("Verify the first Ehrenfest theorem")

    times = sys.dt * np.arange(len(sys.X_average))
    plt.plot(
        times,
        np.gradient(np.array(sys.X_average)[:,0], sys.dt),
        '-r',
        label='$d\\langle x_1 \\rangle / dt$'
    )
    plt.plot(
        times,
        np.array(sys.X_average_RHS)[:,0],
        '--b',
        label='$\\langle p_1 \\rangle$'
    )
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(122)
    plt.title("Verify the second Ehrenfest theorem")

    plt.plot(
        times,
        np.gradient(np.array(sys.P_average)[:,0], sys.dt),
        '-r',
        label='$d\\langle p_1\\rangle / dt$'
    )
    plt.plot(
        times,
        np.array(sys.P_average_RHS)[:,0],
        '--b',
        label='$\\langle -U\'_{x_1} (x_1, x_2, x_3)\\rangle$'
    )
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.show()