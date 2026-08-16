#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Arturo Mena López

Script to simulate the passage of a Gaussian packet wave function through a double slit of finite potential barrier walls of height V0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import time
import datetime

start_time = time.time()
now = datetime.datetime.now()
print(now.strftime("%H:%M:%S"))
def psi0(x, y, x0, y0, sigma=1.5, kx=-1, ky=0):
    
    """
    Proposed wave function for the initial time t=0.
    Initial position: (x0, y0)
    Default parameters:
        - sigma = 0.5 -> Gaussian dispersion.
        - k = 15*np.pi -> Proportional to the momentum.
        
    Note: if Dy=0.1 use np.exp(-1j*k*(x-x0)), if Dy=0.05 use 
          np.exp(1j*k*(x-x0)) so that the particle will move 
          to the right.
    """
    
    return np.exp(-1/2*((x-x0)**2 + (y-y0)**2)/sigma**2)*np.exp(1j*kx*(x-x0))*np.exp(1j*ky*(y-y0))


# =============================================================================
# Parameters
# =============================================================================

L = 32 # Well of width L. Shafts from 0 to +L.
Dy = 0.25 # Spatial step size.
Dt = Dy**2/4 # Temporal step size.
Nx = int(L/Dy) + 1 # Number of points on the x axis.
Ny = int(L/Dy) + 1 # Number of points on the y axis.
Nt = 500 # Number of time steps.
rx = -Dt/(2j*Dy**2) # Constant to simplify expressions.
ry = -Dt/(2j*Dy**2) # Constant to simplify expressions.

# Initial position of the center of the Gaussian wave function.
x0 = 6
y0 = 0

# Parameters of the double slit.
w = 0.6 # Width of the walls of the double slit.
s = 1.0 # Separation between the edges of the slits.
a = 0.2 # Aperture of the slits.

# Indices that parameterize the double slit in the space of points.
# Horizontal axis.
j0 = int(1/(2.5*Dy)*(L-w)) # Left edge.
j1 = int(1/(2.5*Dy)*(L+w)) # Right edge.
# Eje vertical.
i0 = int(1/(2*Dy)*(L+s) + a/Dy) # Lower edge of the lower slit.
i1 = int(1/(2*Dy)*(L+s))        # Upper edge of the lower slit.
i2 = int(1/(2*Dy)*(L-s))        # Lower edge of the upper slit.
i3 = int(1/(2*Dy)*(L-s) - a/Dy) # Upper edge of the upper slit.

# We generate the potential related to the double slit.
#v0 = 200
#v = np.zeros((Ny,Ny), complex) 
#v[0:i3, j0:j1] = v0
#v[i2:i1,j0:j1] = v0
#v[i0:,  j0:j1] = v0
    
Ni = (Nx-2)*(Ny-2)  # Number of unknown factors v[i,j], i = 1,...,Nx-2, j = 1,...,Ny-2


# =============================================================================
# First step: Solve the A·x[n+1] = M·x[n] system for each time step.
# =============================================================================

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

xx = np.linspace(-L/2, L/2, Ny) # Array of spatial points.
yy = np.linspace(-L/2, L/2, Ny) # Array of spatial points.
xx, yy = np.meshgrid(xx, yy)
V = 0*xx + 0*yy#1/((xx/6)**2 + ((yy)/6)**2 + 0.065)
x = np.linspace(-L/2, L/2, Ny-2) # Array of spatial points.
y = np.linspace(-L/2, L/2, Ny-2) # Array of spatial points.
x, y = np.meshgrid(x, y)
psis = [] # To store the wave function at each time step.

v = np.zeros((Ny,Ny), complex)
print(np.shape(v))
v = np.vstack(map(np.ravel, V))
print(np.shape(v))
# =============================================================================
# Second step: Construct the matrices of the system of equations.
# =============================================================================

# Matrices for the Crank-Nicolson calculus. The problem A·x[n+1] = b = M·x[n] will be solved at each time step.
A = np.zeros((Ni,Ni), complex)
M = np.zeros((Ni,Ni), complex)

# We fill the A and M matrices.
for k in range(Ni):     
    
    # k = (i-1)*(Ny-2) + (j-1)
    i = 1 + k//(Ny-2)
    j = 1 + k%(Ny-2)
    
    # Main central diagonal.
    A[k,k] = 1 + 2*rx + 2*ry + 1j*Dt/2*v[i,j]
    M[k,k] = 1 - 2*rx - 2*ry - 1j*Dt/2*v[i,j]
    
    if i != 1: # Lower lone diagonal.
        A[k,(i-2)*(Ny-2)+j-1] = -ry 
        M[k,(i-2)*(Ny-2)+j-1] = ry
        
    if i != Nx-2: # Upper lone diagonal.
        A[k,i*(Ny-2)+j-1] = -ry
        M[k,i*(Ny-2)+j-1] = ry
    
    if j != 1: # Lower main diagonal.
        A[k,k-1] = -rx 
        M[k,k-1] = rx 

    if j != Ny-2: # Upper main diagonal.
        A[k,k+1] = -rx
        M[k,k+1] = rx

Asp = csc_matrix(A)
psi = psi0(x, y, x0, y0) # We initialise the wave function with the Gaussian.
np.shape(psi)
psi[0,:] = psi[-1,:] = psi[:,0] = psi[:,-1] = 0 # The wave function equals 0 at the edges of the simulation box (infinite potential well).
psis.append(np.copy(psi)) # We store the wave function of this time step.

# We solve the matrix system at each time step in order to obtain the wave function.
for i in range(1,Nt):
    psi_vect = psi.reshape((Ni)) # We adjust the shape of the array to generate the matrix b of independent terms.
    b = np.matmul(M,psi_vect) # We calculate the array of independent terms.
    psi_vect = spsolve(Asp,b) # Resolvemos el sistema para este paso temporal.
    psi = psi_vect.reshape((Nx-2,Ny-2)) # Recuperamos la forma del array de la función de onda.
    psis.append(np.copy(psi)) # Save the result.

# We calculate the modulus of the wave function at each time step.
mod_psis = [] # For storing the modulus of the wave function at each time step.
for wavefunc in psis:
    re = np.real(wavefunc) # Real part.
    im = np.imag(wavefunc) # Imaginary part.
    mod = np.sqrt(re**2 + im**2) # We calculate the modulus.
    mod_psis.append(mod) # We save the calculated modulus.
    
## In case there is a need to save memory.
# del psis
# del M
# del psi_vect
# del A
# del Asp
# del b
# del im 
# del re
# del psi

#%%
# =============================================================================
# Third step: We make the animation.
# =============================================================================

fig, ax = plt.subplots(2, figsize=(3,6)) # We create the figure.

img = ax[0].imshow(mod_psis[0], extent=[-L/2,L/2,-L/2,L/2], cmap="Reds", vmin=0, vmax=np.max(mod_psis), zorder=1) # Here the modulus of the 2D wave function shall be represented.
pot = ax[1].contourf(v.real, extent=[-L/2,L/2,-L/2,L/2])
#plt.colorbar(pot)

# We paint the walls of the double slit with rectangles.
slitcolor = "w" # Color of the rectangles.
slitalpha = 0.4 # Transparency of the rectangles.
wall_bottom = Rectangle((j0*Dy,0),     w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha) # (x0, y0), width, height
wall_middle = Rectangle((j0*Dy,i2*Dy), w, (i1-i2)*Dy, color=slitcolor, zorder=50, alpha=slitalpha)
wall_top    = Rectangle((j0*Dy,i0*Dy), w, i3*Dy,      color=slitcolor, zorder=50, alpha=slitalpha)

# We add the rectangular patches to the plot.
#ax[0].add_patch(wall_bottom)
#ax[0].add_patch(wall_middle)
#ax[0].add_patch(wall_top)


# We define the animation function for FuncAnimation.

def animate(i):
    
    """
    Animation function. Paints each frame. Function for Matplotlib's 
    FuncAnimation.
    """
    
    img.set_data(mod_psis[i]) # Fill img with the modulus data of the wave function.
    img.set_zorder(1)
    
    return img, # We return the result ready to use with blit=True.

print("--- %s seconds ---" % (time.time() - start_time))

anim = FuncAnimation(fig, animate, interval=1, frames =np.arange(0,Nt,2), repeat=True, blit=0) # We generate the animation.# Generamos la animación.
anim.save(filename="/Users/hasan/Python Animations/free_particle_fast_2D.gif", writer="pillow",fps=50)
plt.show() # We finally show the animation.