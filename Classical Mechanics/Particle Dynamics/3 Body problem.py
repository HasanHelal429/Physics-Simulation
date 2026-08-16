# Import the nessecary libraries in order to run the program
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Point as pt
import random as rand
from SpacialHashing import *

import matplotlib.animation as animation

# This function uses linear interpolation to find the y value  between two points given the x value
def linInterpColl(z, x, y, vx, vy, j):
    f = y[j] + (z - x[j]) * ((y[j + 1] - y[j]) / (x[j + 1] - x[j]))
    return f

# This function calculates the Distance between any two points
def distanceBetwPts(point1, point2, j1):
    return np.sqrt((point1.x[j1] - point2.x[j1])**2 + (point1.y[j1] - point2.y[j1])**2)

# This function calculates the speed of any point
def speed(point1, i, j):
    return np.sqrt(point1[j].vx[i]**2 + point1[j].vy[i]**2)


# Constants:
collDamp = 0.90
GravC = 0.5

timeRange = 7.5
dt = 0.01
tIteration = int(round(timeRange / dt))

plt.style.use('dark_background')

fig, ax = plt.subplots()
t = np.linspace(0, timeRange, tIteration)

# Mass and Radius of the particles
mass1 = 100
r1 = 0.1

# Initial position and velocity values 
x1 = 0.5
y1 = 4

v1x = 1
v1y = 1

# Number of particles
numP = 4

# Define the point objects for all particles
p1 = [pt.Point(x1, y1, v1x, v1y, mass1, r1)]
for j in range(numP - 1):
    p1.append(pt.Point(rand.random() * 9.7 + 0.1, rand.random() * 9 + 0.1
                       , (rand.random() - 0.5)*5, (rand.random() - 0.5)*5, mass1 + 20, r1))


speed1 = [speed(p1, 0, 0)]

# Global acceleration values
a1x = 0
a1y = 0

# This for loop runs over every frame
for i in range(tIteration + 1):
    # This for loop runs over every point

    coll_Happ = 0
    for j in range(numP):
        for k in range(numP):
            if(j != k):
                a1x = GravC * p1[k].mass * (p1[k].x[i] - p1[j].x[i]) / np.sqrt((p1[k].x[i] - p1[j].x[i])**2 + (p1[k].y[i] - p1[j].y[i])**2)**3
                a1y = GravC * p1[k].mass * (p1[k].y[i] - p1[j].y[i]) / np.sqrt((p1[k].x[i] - p1[j].x[i])**2 + (p1[k].y[i] - p1[j].y[i])**2)**3
        
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

        

        speed1.append(speed(p1, i + 1, j))

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
        for k in range(numP):
            if(distanceBetwPts(p1[j], p1[k], i) < (p1[j].radius + p1[k].radius) and k != j):
                p1[j].vx[i + 1] = p1[j].vx[i] - 2*p1[k].mass*(p1[j].x[i] - p1[k].x[i])*((p1[j].vx[i] - p1[k].vx[i])*(p1[j].x[i] - p1[k].x[i])
                                 + (p1[j].vy[i] - p1[k].vy[i])*(p1[j].y[i] - p1[k].y[i]))/((p1[j].mass + p1[k].mass)*
                                                                                           distanceBetwPts(p1[j], p1[k], i)**2)
                p1[j].vy[i + 1] = p1[j].vy[i] - 2*p1[k].mass*(p1[j].y[i] - p1[k].y[i])*((p1[j].vx[i] - p1[k].vx[i])*(p1[j].x[i] - p1[k].x[i])
                                 + (p1[j].vy[i] - p1[k].vy[i])*(p1[j].y[i] - p1[k].y[i]))/((p1[j].mass + p1[k].mass)*
                                                                                           distanceBetwPts(p1[j], p1[k], i)**2)
                p1[j].x[i + 1] = p1[j].x[i] + dt * p1[j].vx[i + 1] + dt**2 * a1x / 2
                p1[j].y[i + 1] = p1[j].y[i] + dt * p1[j].vy[i + 1] + dt**2 * a1y / 2


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

# This function updates the frame to be displayed in the animation
def update(frame):
    time_text.set_text('Time = %0.2f' % t[frame])
    for j in range(numP):
        scat1[j].set_offsets((p1[j].x[frame],p1[j].y[frame]))

    return scat1,

plt.scatter
ani = animation.FuncAnimation(fig=fig, func=update, frames=tIteration, interval=0.25)

ani.save(filename="/Users/hasan/Python Animations/4_body_orbit_system.gif", writer="pillow",fps=50)

plt.show()