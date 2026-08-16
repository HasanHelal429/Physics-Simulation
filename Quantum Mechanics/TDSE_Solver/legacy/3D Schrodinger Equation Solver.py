import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import scipy.sparse.linalg
import tensorflow as tf
import os
from mpl_toolkits.mplot3d import Axes3D
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
    diag=np.ones([N**2])
    mat=scipy.sparse.spdiags([diag,-2*diag,diag],[-1,0,1],N,N)
    I=scipy.sparse.eye(N)
    L = scipy.sparse.kron(I,scipy.sparse.kron(I,mat)) + scipy.sparse.kron(
        I,scipy.sparse.kron(mat,I)) + scipy.sparse.kron(mat,scipy.sparse.kron(I,I))
    H = -L / (2 * dx**2) #- laplacian / (2 * dy**2)
    if V is not None:
        V = V.flatten()
        H += scipy.sparse.spdiags(V, 0, N**3, N**3)
    return H.tocsc()

def probability_density(psi):
    """Position-space probability density."""
    return psi.real**2 + psi.imag**2

def waveFunctionEigenvectors(H):
    return scipy.sparse.linalg.eigs(H, k=12, which="SM")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = 32
x, dx = np.linspace(-16, 16, N, endpoint=False, retstep=True)
y, dy = np.linspace(-16, 16, N, endpoint=False, retstep=True)
z, dz = np.linspace(-16, 16, N, endpoint=False, retstep=True)

x, y, z = np.meshgrid(x, y, z)

V = (-1 / (((x) / 3)**2 + (y / 3)**2 + (z / 3)**2 + 0.25)**0.5) +  2
H = hamiltonian(N, dx, dy,V=V)

eigenVals, eigenVectors = waveFunctionEigenvectors(H)

psi0 = eigenVectors.T[0].reshape(N,N,N)
energy0 = eigenVals[0].real
psi0[(psi0.real**2 + psi0.imag**2) < 1E-3] = 0

print(np.shape(psi0))
print("--- %s seconds ---" % (time.time() - start_time))

cont = ax.contour3D( probability_density(psi0), cmap='viridis') 
#ax.axes.set_zlim3d(bottom=-2, top=2)
ax.set_title("Energy of WaveFunction: %.2f" % energy0, fontsize=8, pad=2)
fig.colorbar(cont, ax=ax)

#plt.savefig(fname="/Users/hasan/Python Animations/coulomb_potential_2D_basis_further_extensions.png")

plt.show()