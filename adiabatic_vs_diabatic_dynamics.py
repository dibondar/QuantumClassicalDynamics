"""
Demonstration of the difference between adiabatic (slow) and diabatic (fast) evolution.
In particular, we will illustrate the adiabatic theorem that states:
A physical system remains in its instantaneous eigenstate if a given perturbation is acting slowly enough.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # tools for creating animation
from imag_time_propagation import ImgTimePropagation
from numba import njit


class DynamicVisualized:
    """
    We bundle creation of animation into this class
    """
    def __init__(self, fig):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        """
        #################################################################
        #
        #   Initialize systems
        #
        #################################################################

        # common quantum system parameters
        @njit
        def k(p, t=0.):
            """
            Non-relativistic kinetic energy
            """
            return 0.5 * p ** 2

        self.qsys_params = dict(
            t=-7.,
            dt=0.005,
            x_grid_dim=1024,
            x_amplitude=10.,
            k=k,
        )

        @njit
        def v1(x, t=-7.):
            """
            Slowly changing potential energy
            """
            return 0.01 * (1. - 0.95 / (1. + np.exp(-0.2 * t))) * x ** 4

        # initialize adiabatic system (i.e, with slow time dependence)
        self.adiabatic_sys = ImgTimePropagation(
                v=v1,
                **self.qsys_params
        )

        @njit
        def v2(x, t=-7.):
            """
            Rapidly changing potential energy
            """
            return 0.01 * (1. - 0.95 / (1. + np.exp(-5. * t))) * x ** 4

        # initialize diabatic system (i.e, with fast time dependence)
        self.diabatic_sys = ImgTimePropagation(
                v=v2,
                **self.qsys_params
        )

        #################################################################
        #
        #   Initialize plotting facility
        #
        #################################################################

        self.fig = fig

        # plotting axis limits
        xmin = self.diabatic_sys.x.min()
        xmax = self.diabatic_sys.x.max()
        ymin = 1e-10
        ymax = 1e2

        # prepare for plotting diabatic dynamics
        adiabatic_ax = self.fig.add_subplot(121)
        adiabatic_ax.set_title("Adiabatic evolution")
        self.adiabatic_line, = adiabatic_ax.semilogy([], [], 'r-', label='exact wave function')
        self.ad_instant_eigns_line, = adiabatic_ax.semilogy([], [], 'b--', label='instantaneous eigenstate')
        adiabatic_ax.set_xlim(xmin, xmax)
        adiabatic_ax.set_ylim(ymin, ymax)
        adiabatic_ax.legend()
        adiabatic_ax.set_xlabel("$x$ (a.u.)")
        adiabatic_ax.set_ylabel("probability density")

        # prepare for plotting diabatic dynamics
        diabatic_ax = self.fig.add_subplot(122)
        diabatic_ax.set_title("Diabatic evolution")
        self.diabatic_line, = diabatic_ax.semilogy([], [], 'r-', label='exact wave function')
        self.d_instant_eigns_line, = diabatic_ax.semilogy([], [], 'b--', label='instantaneous eigenstate')
        diabatic_ax.set_xlim(xmin, xmax)
        diabatic_ax.set_ylim(ymin, ymax)
        diabatic_ax.legend()
        diabatic_ax.set_xlabel("$x$ (a.u.)")
        #diabatic_ax.set_ylabel("probability density")

        # Bundle all graphical objects
        self.lines = (
            self.ad_instant_eigns_line,
            self.adiabatic_line,
            self.d_instant_eigns_line,
            self.diabatic_line
        )

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: line objects
        """
        # find instantaneous ground states
        ad_ground_state = self.adiabatic_sys.get_stationary_states(1, nsteps=5000).stationary_states[0]
        d_ground_state = self.diabatic_sys.get_stationary_states(1, nsteps=5000).stationary_states[0]

        self.ad_instant_eigns_line.set_data(
            self.adiabatic_sys.x,
            np.abs(ad_ground_state) ** 2
        )

        self.d_instant_eigns_line.set_data(
            self.diabatic_sys.x,
            np.abs(d_ground_state) ** 2
        )

        if frame_num == 0:
            # this is the first frame then, set the initial condition
            self.adiabatic_sys.set_wavefunction(ad_ground_state)
            self.diabatic_sys.set_wavefunction(d_ground_state)
        else:
            # propagate
            self.adiabatic_sys.propagate(100)
            self.diabatic_sys.propagate(100)

        # update plots
        self.adiabatic_line.set_data(
            self.adiabatic_sys.x,
            np.abs(self.adiabatic_sys.wavefunction)**2
        )
        self.diabatic_line.set_data(
            self.diabatic_sys.x,
            np.abs(self.diabatic_sys.wavefunction)**2
        )

        return self.lines

fig = plt.gcf()
visualizer = DynamicVisualized(fig)
animation = FuncAnimation(
    fig, visualizer, frames=np.arange(100), repeat=True, blit=True
)
plt.show()

# If you want to make a movie, comment "plt.show()" out and uncomment the lines bellow

# Set up formatting for the movie files
# writer = writers['mencoder'](fps=10, metadata=dict(artist='a good student'), bitrate=-1)

# Save animation into the file
# animation.save('2D_Schrodinger.mp4', writer=writer)
