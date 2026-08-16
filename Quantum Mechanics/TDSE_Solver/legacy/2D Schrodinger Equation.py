import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import tensorflow as tf
import os
from scipy.optimize import root_scalar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start_time = time.time()

def hamiltonian(N, dx, dy, V=None):
    """Returns Hamiltonian using finite differences.

    Args:
        N (int): Number of grid points.
        dx (float): Grid spacing.
        V (array-like): Potential. Must have shape (N,).
            Default is a zero potential everywhere.

    Returns:
        Hamiltonian as a sparse matrix with shape (N, N).
    """
    diag=np.ones([N*N])
    mat=scipy.sparse.spdiags([diag,-2*diag,diag],[-1,0,1],N,N)
    I=scipy.sparse.eye(N)
    L = scipy.sparse.kron(I,mat,format='csr')+scipy.sparse.kron(mat,I)
    H = -L / (2 * dx**2) #- laplacian / (2 * dy**2)
    if V is not None:
        V = V.flatten()
        H += scipy.sparse.spdiags(V, 0, N**2, N**2)
    return H.tocsc()

def time_evolution_operator(H, dt):
    """Time evolution operator given a Hamiltonian and time step."""
    M = scipy.sparse.csr_matrix.todense(-1j * H * dt)
    U = tf.linalg.expm(M).numpy()
    U[(U.real**2 + U.imag**2) < 1E-10] = 0
    return scipy.sparse.csc_matrix(U)


def simulate(psi, H, dt):
    """Generates wavefunction and time at the next time step."""
    U = time_evolution_operator(H, dt)
    print("--- %s seconds ---" % (time.time() - start_time))
    t = 0
    while True:
        yield psi, t * dt
        psiFlat = psi.flatten()
        psi = (U @ psiFlat).reshape(N,N)
        t += 1

def probability_density(psi):
    """Position-space probability density."""
    return psi.real**2 + psi.imag**2

def gaussian_wavepacket(x, y, x0, y0, sigma0, p0x, p0y):
    """Gaussian wavepacket at x0 +/- sigma0, with average momentum, p0."""
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A * np.exp(1j*p0x*x + 1j*p0y*y - ((x - x0)/(2 * sigma0))**2 - ((y - y0)/(2 * sigma0))**2)

def rectangular_potential_barrier(x, V0, a):
    """Rectangular potential barrier of height V0 and width a."""
    return np.where((0 <= x) & (x < a), V0, 0.0)

def transmission_probability(E, V0, a):
    """Transmission probability of through a rectangular potential barrier."""
    k = (V0 * np.sinh(a * np.sqrt(2 * (V0 - E))))**2 / (4 * E * (V0 - E))
    return 1 / (1 + k)

def double_slit_potential_barrier(x, y, V0, a, w, d):
    """Rectangular potential barrier of height V0 and width a."""
    return np.where((4 <= x) & (x < a + 4) & (~ (((y >= d/2) & (y < d/2 + w)) | ((y <= -d/2) & (y > -d/2 - w)))), V0, 0.0)

plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
time_text = ax.text(0.02, 1.05, '', transform=ax.transAxes)

tIteration = 300
N = 128
x, dx = np.linspace(-16, 16, N, endpoint=False, retstep=True)
y, dy = np.linspace(-16, 16, N, endpoint=False, retstep=True)

x, y = np.meshgrid(x, y)

w = 2
d = 4
T, r = 0.6, 3/4
k1 = root_scalar(lambda a: transmission_probability(0.5*r, 0.5, a) - T,
                 bracket=(0.0, 10.0)).root

a = 1.25
V0 = ((k1 / a)**2) / 2
E = r * V0

psi0 = gaussian_wavepacket(x, y, x0=12.0, y0=0.0, sigma0=1.5, p0x=-1.0, p0y=0.0)

V = double_slit_potential_barrier(x, y, V0, a, w, d)
H = hamiltonian(N, dx, dy,V=V)
print(np.shape(H))

psiPlot = plt.imshow(probability_density(psi0), cmap ='Reds', extent =[x.min(), x.max(), y.min(), y.max()])

sim = simulate(psi0, H, dt=1.0)

#t = range(1000)
def animate(i):
    global t
    a=psiPlot.get_array()
    a= probability_density(next(sim)[0]) 
    psiPlot.set_array(a)
    #time_text.set_text('Time = %0.2f' % t[i])
    return [psiPlot]

ani = animation.FuncAnimation(
    fig, animate, interval=75, blit=True, save_count=tIteration,repeat=False)

ani.save(filename="/Users/hasan/Python Animations/double_slit_2D.gif", writer="pillow",fps=30)

#plt.show()
