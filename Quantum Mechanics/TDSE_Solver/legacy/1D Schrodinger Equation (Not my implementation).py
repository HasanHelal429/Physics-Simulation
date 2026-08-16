import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root_scalar

def hamiltonian(N, dx, V=None):
    """Returns Hamiltonian using finite differences.

    Args:
        N (int): Number of grid points.
        dx (float): Grid spacing.
        V (array-like): Potential. Must have shape (N,).
            Default is a zero potential everywhere.

    Returns:
        Hamiltonian as a sparse matrix with shape (N, N).
    """
    L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))
    H = -L / (2 * dx**2)
    if V is not None:
        H += scipy.sparse.spdiags(V, 0, N, N)
    return H.tocsc()

def time_evolution_operator(H, dt):
    """Time evolution operator given a Hamiltonian and time step."""
    U = scipy.sparse.linalg.expm(-1j * H * dt).toarray()
    U[(U.real**2 + U.imag**2) < 1E-10] = 0
    return scipy.sparse.csc_matrix(U)


def simulate(psi, H, dt):
    """Generates wavefunction and time at the next time step."""
    U = time_evolution_operator(H, dt)
    t = 0
    while True:
        yield psi, t * dt
        psi = U @ psi
        t += 1

def probability_density(psi):
    """Position-space probability density."""
    return psi.real**2 + psi.imag**2

def gaussian_wavepacket(x, x0, sigma0, p0):
    """Gaussian wavepacket at x0 +/- sigma0, with average momentum, p0."""
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A * np.exp(1j*p0*x - ((x - x0)/(2 * sigma0))**2)

def rectangular_potential_barrier(x, V0, a):
    """Rectangular potential barrier of height V0 and width a."""
    return np.where((0 <= x) & (x < a), V0, 0.0)


def transmission_probability(E, V0, a):
    """Transmission probability of through a rectangular potential barrier."""
    k = (V0 * np.sinh(a * np.sqrt(2 * (V0 - E))))**2 / (4 * E * (V0 - E))
    return 1 / (1 + k)

def Potential(x, z=10, k=1, e=1, v0 = 0, a0=5):
    potential = np.zeros(len(x))
    for i in range(len(x)):
        if(x[i]<a0):
            potential[i] = v0
        else:
            potential[i] = 2*z*k*e**2/x[i]
    return potential

fig, ax = plt.subplots()

tIteration = 500
N = 256
x, dx = np.linspace(0, 10, N, endpoint=False, retstep=True)

T, r = 0.20, 3/4
k1 = root_scalar(lambda a: transmission_probability(0.5*r, 0.5, a) - T,
                 bracket=(0.0, 10.0)).root

a = 1.25
V0 = ((k1 / a)**2) / 2
E = 10

x0, sigma0, p0 = 1, 0.5, 0.5
psi0 = gaussian_wavepacket(x, x0=x0, sigma0=sigma0, p0=p0)

V = Potential(x)
H = hamiltonian(N, dx, V=V)

plt.plot(x,probability_density(psi0))
plt.plot(x,V)
line, = ax.plot(x, probability_density(psi0))
linePotential = ax.plot(x, V)
sim = simulate(psi0, H, dt=1.0)

ax.set( ylim=[0, 0.5], xlabel='X (m)', ylabel='Y (m)')

def animate(i):
    line.set_ydata(probability_density(next(sim)[0]))  # update the data.
    return line,

ani = animation.FuncAnimation(
    fig, animate, blit=True, frames=tIteration, interval = 20)

#ani.save(filename="/Users/hasan/Python Animations/Quantum Tunneling 1D.gif", writer="pillow",fps=30)

plt.show()