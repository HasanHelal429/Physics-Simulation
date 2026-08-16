
# Implementation of matplotlib function 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm 
import matplotlib.animation as animation
import Point as pt
import random as rand
      
 # Define vector field
def vField(x, p1,i):
    u = (x[0] - p1.x[i]) / (0.05 + np.sqrt((x[0] - p1.x[i])**2 + (x[1] - p1.y[i])**2)**3)
    v = (x[1] - p1.y[i]) / (0.05 + np.sqrt((x[0] - p1.x[i])**2 + (x[1] - p1.y[i])**2)**3)
    return [u,v]

# This function calculates the Distance between any two points
def distanceBetwPts(point1, point2, j1):
    return np.sqrt((point1.x[j1] - point2.x[j1])**2 + (point1.y[j1] - point2.y[j1])**2)
    
# Constants:
collDamp = 0.90

timeRange = 7.5
dt = 0.01
tIteration = int(round(timeRange / dt))

plt.style.use('dark_background')

t = np.linspace(0, timeRange, tIteration)
fig, ax = plt.subplots()

# Mass and Radius of the particles
mass1 = 10
r1 = 0.025

# Initial position and velocity values 
x1 = 0.5
y1 = 4

v1x = 2
v1y = 0

# Number of particles
numP = 1
numCollisions = 0

# Define the point objects for all particles
p1 = [pt.Point(x1, y1, v1x, v1y, mass1, r1)]
for j in range(numP - 1):
    p1.append(pt.Point(rand.random() * 9.7 + 0.1, rand.random() * 9 + 0.1, v1x, v1y, mass1 + 20, r1))

# Global acceleration values
a1x = 0
a1y = 0

# This for loop runs over every frame
for i in range(tIteration + 1):

    coll_Happ = 0
    for j in range(numP):
        # Iterate the position based on velocity
        x1temp = p1[j].x[i] + dt * p1[j].vx[i] + dt**2 * a1x / 2
        y1temp = p1[j].y[i] + dt * p1[j].vy[i] + dt**2 * a1y / 2

        p1[j].x.append(x1temp)
        p1[j].y.append(y1temp)

        # Iterate the velocity based on position
        vx1temp = p1[j].vx[i] + dt * a1x
        vy1temp = p1[j].vy[i] + dt * a1y

        p1[j].vx.append(vx1temp)
        p1[j].vy.append(vy1temp)


        # Collision detection between particle and floor
        if p1[j].y[i + 1] < 0:
            f1 = p1[j].x[i] + (0 - p1[j].y[i]) * ((p1[j].x[i + 1] - p1[j].x[i]) /  (p1[j].y[i + 1] - p1[j].y[i]))
            p1[j].y[i + 1] = 0
            p1[j].x[i + 1] = f1

            dtInt = (p1[j].x[i + 1] - p1[j].x[i]) / p1[j].vx[i]

            t = np.insert(t, i + 1, t[i] + dtInt)

            p1[j].vx[i + 1] = p1[j].vx[i]
            p1[j].vy[i + 1] = -(p1[j].vy[i + 1] * (dtInt / dt) + p1[j].vy[i] * (1 - (dtInt / dt))) * collDamp

            for c in range(tIteration):
                if(c > i + 1):
                    t[c] += dtInt

        # Collision detection between particle and ceiling
        if p1[j].y[i + 1] > 10:
            f1 = p1[j].x[i] + (10 - p1[j].y[i]) * ((p1[j].x[i + 1] - p1[j].x[i]) /  (p1[j].y[i + 1] - p1[j].y[i]))
            p1[j].y[i + 1] = 10
            p1[j].x[i + 1] = f1

            dtInt = (p1[j].x[i + 1] - p1[j].x[i]) / p1[j].vx[i]

            t = np.insert(t, i + 1, t[i] + dtInt)

            p1[j].vx[i + 1] = p1[j].vx[i]
            p1[j].vy[i + 1] = -(p1[j].vy[i + 1] * (dtInt / dt) + p1[j].vy[i] * (1 - (dtInt / dt))) * collDamp

            for c in range(tIteration):
                if(c > i + 1):
                    t[c] += dtInt

        # Collision detection between particle and right wall
        if p1[j].x[i + 1] > 10:
            f1 = p1[j].y[i] + (10 - p1[j].x[i]) * (( p1[j].y[i + 1] - p1[j].y[i]) / (p1[j].x[i + 1] - p1[j].x[i]))
            p1[j].y[i + 1] = f1
            p1[j].x[i + 1] = 10

            dtInt = (p1[j].x[i + 1] - p1[j].x[i]) / p1[j].vx[i]

            t = np.insert(t, i + 1, t[i] + dtInt)
            
            
            p1[j].vx[i + 1] = -(p1[j].vx[i + 1] * (dtInt / dt) + p1[j].vx[i] * (1 - (dtInt / dt)))
            p1[j].vy[i + 1] = p1[j].vy[i]
            

            for b in t:
                if(b > i + 1):
                    t[b] += dtInt
            
        # Collision detection between particle and left wall
        if p1[j].x[i + 1] < 0:
            f1 = p1[j].y[i] + (0 - p1[j].x[i]) * ((p1[j].y[i + 1] - p1[j].y[i]) / (p1[j].x[i + 1] - p1[j].x[i]))
            p1[j].y[i + 1] = f1
            p1[j].x[i + 1] = 0

            dtInt = (p1[j].x[i + 1] - p1[j].x[i]) / p1[j].vx[i]

            t = np.insert(t, i + 1, t[i] + dtInt)
            
            p1[j].vx[i + 1] = -(p1[j].vx[i + 1] * (dtInt / dt) + p1[j].vx[i] * (1 - (dtInt / dt)))
            p1[j].vy[i + 1] = p1[j].vy[i]
            

            for b in t:
                if(b > i + 1):
                    t[b] += dtInt
        
        # Collision detection between particles
        for q in range(7):
            k = j - 3 + q
            if(k >= 0 and k < numP):
                if(distanceBetwPts(p1[j], p1[k], i) < (p1[j].radius + p1[k].radius) and k != j):
                    numCollisions += 1
                    p1[j].vx[i + 1] = p1[j].vx[i] - 2*p1[k].mass*(p1[j].x[i] - p1[k].x[i])*((p1[j].vx[i] - p1[k].vx[i])*(p1[j].x[i] - p1[k].x[i])
                                    + (p1[j].vy[i] - p1[k].vy[i])*(p1[j].y[i] - p1[k].y[i]))/((p1[j].mass + p1[k].mass)*
                                                                                            distanceBetwPts(p1[j], p1[k], i)**2)
                    p1[j].vy[i + 1] = p1[j].vy[i] - 2*p1[k].mass*(p1[j].y[i] - p1[k].y[i])*((p1[j].vx[i] - p1[k].vx[i])*(p1[j].x[i] - p1[k].x[i])
                                    + (p1[j].vy[i] - p1[k].vy[i])*(p1[j].y[i] - p1[k].y[i]))/((p1[j].mass + p1[k].mass)*
                                                                                            distanceBetwPts(p1[j], p1[k], i)**2)
                    p1[j].x[i + 1] = p1[j].x[i] + dt * p1[j].vx[i + 1] + dt**2 * a1x / 2
                    p1[j].y[i + 1] = p1[j].y[i] + dt * p1[j].vy[i + 1] + dt**2 * a1y / 2


X, Y = np.mgrid[0:10:30j,0:10:30j]
U, V = vField([X,Y], p1[0], 0)
q = ax.quiver(X, Y, U, V, color = 'w', scale = 50)

# Define the scatter plot array
scat1 = [ax.scatter(p1[0].x[0], p1[0].y[0], c="b", s=5
                    , label=f'v1 = ({p1[0].vx[0]}, {p1[0].vy[0]}) m/s'
                    )]
for j in range(numP - 1):
    scat1.append(ax.scatter(p1[j + 1].x[0], p1[j + 1].y[0], c='b', s=5 
                            #, label=f'v1 = ({p1[j + 1].vx[0]}, {p1[j + 1].vy[0]}) m/s'
                            ))

# Define the limits of the plot
ax.set(xlim=[0, 10], ylim=[0, 10], xlabel='X (m)', ylabel='Y (m)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()

def animate(i):
    time_text.set_text('Time = %0.2f' % t[i])
    U, V = vField([X,Y], p1[0], i)
    q.set_UVC(U, V)
    for j in range(numP):
        scat1[j].set_offsets((p1[j].x[i],p1[j].y[i]))
    return scat1,

ani = animation.FuncAnimation(fig, animate, frames=tIteration, repeat=True, interval=0.25)
#ani.save(filename="/Users/hasan/Python Animations/Electric Field of Point Particle.gif", writer="pillow",fps=50)

plt.show()