# Implementation of matplotlib function 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm 
import matplotlib.animation as animation
import Point1 as pt
import random as rand
      
 # Define vector field
def vField(x,t,a):
    u = np.sin(x[1])
    v = np.cos(x[0]-10*a/(2*np.pi))
    return [u,v]
    
# Constants:
collDamp = 0.90

timeRange = 7.5
dt = 0.01
tIteration = int(round(timeRange / dt))

plt.style.use('dark_background')


t = np.linspace(0, timeRange, tIteration)

fig, ax = plt.subplots()
X, Y = np.mgrid[-2:2:20j,-2:2:20j]
U, V = vField([X,Y],0, t[0])
q = ax.quiver(X, Y, U, V, color = 'w')



def animate(i):
    U, V = vField([X,Y],0, t[i])
    q.set_UVC(U, V)

ani = animation.FuncAnimation(fig, animate, frames=tIteration, repeat=True, interval=0.25)
#ani.save(filename="/Users/hasan/Python Animations/Vector Field Animation.gif", writer="pillow",fps=50)
plt.show()