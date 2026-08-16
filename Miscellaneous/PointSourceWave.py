# Implementation of matplotlib function 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm 
import matplotlib.animation as animation
      
#def disFromOrig()

dx, dy = 0.015, 0.05
y, x = np.mgrid[slice(-4, 4 + dy, dy), 
                slice(-4, 4 + dx, dx)] 
z = np.sin(4*np.sqrt(x**2+y**2))*np.exp(-np.sqrt(x**2+y**2))
z = z[:-1, :-1] 
z_min, z_max = -np.abs(z).max(), np.abs(z).max() 
  
fig = plt.figure()

c = plt.imshow(z, cmap ='coolwarm', vmin = z_min, vmax = z_max, 
                 extent =[x.min(), x.max(), y.min(), y.max()]) 
plt.colorbar(c) 
  
#plt.title('Sine wave indefinite')

# initialization function: plot the background of each frame
def init():
    c.set_data(z)
    return [c]

# animation function.  This is called sequentially
def animate(i):
    a=c.get_array()
    a= np.sin((4*np.sqrt(x**2+y**2))-0.05*i)/np.sqrt(x**2+y**2+1.5)     # exponential decay of the values
    c.set_array(a)
    return [c]

ani = animation.FuncAnimation(fig=fig, func=animate, frames=126, interval=0.25,
                              init_func=init)

#ani.save(filename="/Users/hasan/Python Animations/point_source_light_wave.gif", writer="pillow")

plt.show()