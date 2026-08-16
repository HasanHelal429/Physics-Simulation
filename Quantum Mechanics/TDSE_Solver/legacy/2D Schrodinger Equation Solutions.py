import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import scipy.sparse.linalg
import tensorflow as tf
import os
from mpl_toolkits.mplot3d import axes3d
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
    L = scipy.sparse.kron(I,mat)+scipy.sparse.kron(mat,I)
    H = -L / (2 * dx**2) #- laplacian / (2 * dy**2)
    if V is not None:
        V = V.flatten()
        H += scipy.sparse.spdiags(V, 0, N**2, N**2)
    return H.tocsc()

def probability_density(psi):
    """Position-space probability density."""
    return psi.real**2 + psi.imag**2

def waveFunctionEigenvectors(H):
    return scipy.sparse.linalg.eigs(H, k=12, which="SM")

def rectangular_potential_barrier(x, V0, a):
    """Rectangular potential barrier of height V0 and width a."""
    return np.where((0 <= x) & (x < a), V0, 0.0)

def transmission_probability(E, V0, a):
    """Transmission probability of through a rectangular potential barrier."""
    k = (V0 * np.sinh(a * np.sqrt(2 * (V0 - E))))**2 / (4 * E * (V0 - E))
    return 1 / (1 + k)

def double_slit_potential_barrier(x, y, V0, a, w, d):
    """Rectangular potential barrier of height V0 and width a."""
    return np.where((0 <= x) & (x < a) & (~ (((y >= d/2) & (y < d/2 + w)) | ((y <= -d/2) & (y > -d/2 - w)))), V0, 0.0)

fig, ax = plt.subplots(2,2)

N = 128
x, dx = np.linspace(-16, 16, N, endpoint=False, retstep=True)
y, dy = np.linspace(-16, 16, N, endpoint=False, retstep=True)

x, y = np.meshgrid(x, y)

T, r = 0.1, 3/4
k1 = root_scalar(lambda a: transmission_probability(0.5*r, 0.5, a) - T,
                 bracket=(0.0, 10.0)).root

w = 2
d = 4
a = 1.25
V0 = ((k1 / a)**2) / 2
E = r * V0

V = double_slit_potential_barrier(x, y, V0, a, w, d)
H = hamiltonian(N, dx, dy,V=V)

eigenVals, eigenVectors = waveFunctionEigenvectors(H)

psi0 = eigenVectors.T[0].reshape(N,N)
energy0 = eigenVals[0].real

psi1 = eigenVectors.T[1].reshape(N,N)
energy1 = eigenVals[1].real

psi2 = eigenVectors.T[2].reshape(N,N)
energy2 = eigenVals[2].real

print(np.shape(psi0))
print("--- %s seconds ---" % (time.time() - start_time))

ax[0, 0].imshow(probability_density(psi0), cmap ='Reds', extent =[x.min(), x.max(), y.min(), y.max()])
ax[0, 0].set_title("Energy of WaveFunction: %.2f" % energy0, fontsize=8, pad=2)

ax[0, 1].imshow(probability_density(psi1), cmap ='Reds', extent =[x.min(), x.max(), y.min(), y.max()])
ax[0, 1].set_title("Energy of WaveFunction: %.2f" % energy1, fontsize=8, pad=2)

#ax[1,0] = fig.add_subplot(111, projection="3d")
ax[1, 0].imshow(probability_density(psi2), cmap ='Reds', extent =[x.min(), x.max(), y.min(), y.max()])
ax[1, 0].set_title("Energy of WaveFunction: %.2f" % energy2, fontsize=8, pad=2)
#ax[1,0].plot_surface(x, y, probability_density(psi0), cmap="autumn_r", lw=0.5, rstride=1, cstride=1)

vPot = ax[1, 1].contourf(V, extent =[x.min(), x.max(), y.min(), y.max()])
ax[1, 1].set_title("Potential", fontsize=8, pad=2)
plt.colorbar(vPot)

#plt.savefig(fname="/Users/hasan/Python Animations/finite_wall_2D_basis.png")

plt.show()