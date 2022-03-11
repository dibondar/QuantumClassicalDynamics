from split_op_schrodinger2D import *

# load tools for creating animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

# Use the documentation string for the developed class
print(SplitOpSchrodinger2D.__doc__)


class VisualizeDynamics2D:
    """
    Class to visualize the wave function dynamics in 2D.
    """
    def __init__(self, fig):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        """
        #  Initialize systems
        self.set_quantum_sys()

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        self.fig = fig

        ax = fig.add_subplot(111)

        ax.set_title('Wavefunction density, $| \\Psi(t, x, s) |^2$')
        extent=[
            self.quant_sys.x2.min(),
            self.quant_sys.x2.max(),
            self.quant_sys.x1.min(),
            self.quant_sys.x1.max()
        ]
        self.img = ax.imshow([[]], extent=extent, origin='lower')

        self.fig.colorbar(self.img)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$t$ (a.u.)')

    def set_quantum_sys(self):
        """
        Initialize quantum propagator
        :param self:
        :return:
        """

        omega = 3.

        @njit
        def v(t, x, s=0.):
            """
            Potential energy
            """
            return 0.5 * omega ** 2 * x ** 2 + 5 * x * np.sin(t)

        @njit
        def diff_v_t(t, x, s=0.):
            """
            the derivative of the potential energy for Ehrenfest theorem evaluation
            """
            return 5 * np.cos(t) * x

        @njit
        def diff_v_x(t, x, s=0.):
            """
            the derivative of the potential energy for Ehrenfest theorem evaluation
            """
            return (omega) ** 2 * x + 5 * np.sin(t)

        @njit
        def k(E, p, s=0.):
            """
            Non-relativistic kinetic energy
            """
            return E + 0.5 * p ** 2

        @njit
        def diff_k_E(E, p, s=0.):
            """
            the derivative of the kinetic energy for Ehrenfest theorem evaluation
            """
            return 1

        @njit
        def diff_k_p(E, p, s=0.):
            """
            the derivative of the kinetic energy for Ehrenfest theorem evaluation
            """
            return p

        self.quant_sys = SplitOpSchrodinger2D(
            t=0.,
            dt=0.005,
            x1_grid_dim=256,
            x1_amplitude=5.,
            x2_grid_dim=256,
            x2_amplitude=5.,

            # kinetic energy part of the hamiltonian
            k=k,

            # these functions are used for evaluating the Ehrenfest theorems
            diff_k_p1=diff_k_E,
            diff_k_p2=diff_k_p,

            # potential energy part of the hamiltonian
            v=v,

            # these functions are used for evaluating the Ehrenfest theorems
            diff_v_x1=diff_v_t,
            diff_v_x2=diff_v_x,
        )

        # set initial condition
        self.quant_sys.set_wavefunction(
            lambda t, x: np.exp(-15 * (t - 0.9 * self.quant_sys.x1.min()) ** 2 - (x) ** 2)
        )

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        # propagate and set the density
        self.img.set_array(
            np.abs(self.quant_sys.propagate(10)) ** 2
        )
        return self.img,

fig = plt.gcf()
visualizer = VisualizeDynamics2D(fig)
animation = FuncAnimation(
    fig, visualizer, frames=np.arange(100), repeat=True, blit=True
)

plt.show()

# If you want to make a movie, comment "plt.show()" out and uncomment the lines bellow

# Set up formatting for the movie files
#   writer = writers['mencoder'](fps=10, metadata=dict(artist='a good student'), bitrate=-1)

# Save animation into the file
#   animation.save('2D_Schrodinger.mp4', writer=writer)

# extract the reference to quantum system
quant_sys = visualizer.quant_sys

# Analyze how well the energy was preserved
h = np.array(quant_sys.hamiltonian_average)
print(
    "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
        100. * (1. - h.min()/h.max())
    )
)

#################################################################
#
# Plot the Ehrenfest theorems after the animation is over
#
#################################################################

# generate time step grid
dt = quant_sys.dt
times = np.arange(dt, dt + dt*len(quant_sys.x1_average), dt)

plt.subplot(121)
plt.title("The first Ehrenfest theorem verification")

plt.plot(
    times,
    np.gradient(quant_sys.x1_average, dt),
    'r-',
    label='$d\\langle \\hat{x}_1 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.x1_average_rhs,
    'b--', label='$\\langle \\hat{p}_1 \\rangle$'
)

plt.plot(
    times,
    np.gradient(quant_sys.x2_average, dt),
    'g-',
    label='$d\\langle \\hat{x}_2 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.x2_average_rhs,
    'k--',
    label='$\\langle \\hat{p}_2 \\rangle$'
)

plt.legend()
plt.xlabel('time $t$ (a.u.)')

plt.subplot(122)
plt.title("The second Ehrenfest theorem verification")

plt.plot(
    times,
    np.gradient(quant_sys.p1_average, dt),
    'r-',
    label='$d\\langle \\hat{p}_1 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.p1_average_rhs,
    'b--',
    label='$\\langle -\\partial\\hat{V}/\\partial\\hat{x}_1 \\rangle$'
)

plt.plot(
    times,
    np.gradient(quant_sys.p2_average, dt),
    'g-',
    label='$d\\langle \\hat{p}_2 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.p2_average_rhs,
    'k--',
    label='$\\langle -\\partial\\hat{V}/\\partial\\hat{x}_2 \\rangle$'
)

plt.legend()
plt.xlabel('time $t$ (a.u.)')

plt.show()