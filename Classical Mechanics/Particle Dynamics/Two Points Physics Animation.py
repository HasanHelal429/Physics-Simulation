import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Point as pt

import matplotlib.animation as animation

def linInterpColl(z, x, y, vx, vy, j):
    f = y[j] + (z - x[j]) * ((y[j + 1] - y[j]) / (x[j + 1] - x[j]))
    return f

def distanceBetw(x, y, j):
    return np.sqrt((x[j + 1] - x[j])**2 + (y[j + 1] - y[j])**2)

def speed(vx, vy, j):
    return np.sqrt(vx[j]**2 + vy[j]**2)

# Constants:
collDamp = 0.95

timeRange = 10
dt = 0.01
tIteration = int(round(timeRange / dt))

fig, ax = plt.subplots()
t = np.linspace(0, timeRange, tIteration)

x1 = 0 
y1 = 4

v1x = 1
v1y = 3

x2 = 3 
y2 = 0

v2x = 2
v2y = 10

p1 = pt.Point(x1, y1, v1x, v1y)
p2 = pt.Point(x2, y2, v2x, v2y)

speed1 = [speed(p1.vx, p1.vy, 0)]
speed2 = [speed(p2.vx, p2.vy, 0)]

a1x = 0
a1y = -9.81

a2x = -3
a2y = -9.81


for i in range(tIteration + 1):
    #iterate the position based on velocity
    x1temp = p1.x[i] + dt * p1.vx[i] + dt**2 * a1x / 2
    y1temp = p1.y[i] + dt * p1.vy[i] + dt**2 * a1y / 2

    x2temp = p2.x[i] + dt * p2.vx[i] + dt**2 * a2x / 2
    y2temp = p2.y[i] + dt * p2.vy[i] + dt**2 * a2y / 2

    p1.x.append(x1temp)
    p1.y.append(y1temp)

    p2.x.append(x2temp)
    p2.y.append(y2temp)

    d1 = distanceBetw(p1.x, p1.y, i)
    d2 = distanceBetw(p2.x, p2.y, i)

    #iterate the velocity based on position
    vx1temp = p1.vx[i] + dt * a1x
    vy1temp = p1.vy[i] + dt * a1y
    vx2temp = p2.vx[i] + dt * a2x
    vy2temp = p2.vy[i] + dt * a2y

    p1.vx.append(vx1temp)
    p1.vy.append(vy1temp)
    p2.vx.append(vx2temp)
    p2.vy.append(vy2temp)

    speed1.append(speed(p1.vx, p1.vy, i + 1))
    speed2.append(speed(p2.vx, p2.vy, i + 1))
    
    # if (p1.vy[i] < 0.1) & (p1.vy[i] > -0.1):
    #     print(9.81 * p1.y[i] + p1.vx[i]**2 / 2)

    #Collision detection for p1
    if p1.y[i + 1] < 0:
        f1 = p1.x[i] + (0 - p1.y[i]) * ((p1.x[i + 1] - p1.x[i]) /  (p1.y[i + 1] - p1.y[i]))
        p1.y[i + 1] = 0
        p1.x[i + 1] = f1

        speedi = speed(p1.vx, p1.vy, i)
        speedf = speed(p1.vx, p1.vy, i + 1)
        distanceInt = distanceBetw(p1.x, p1.y, i)
        dtInt = (p1.x[i + 1] - p1.x[i]) / p1.vx[i]

        t[i + 1] = t[i] + dtInt

        p1.vx[i + 1] = p1.vx[i]
        p1.vy[i + 1] = -(p1.vy[i + 1] * (dtInt / dt) + p1.vy[i] * (1 - (dtInt / dt))) * collDamp

        for c in range(tIteration):
            if(c > i + 1):
                t[c] -= dt - dtInt

    if p1.x[i + 1] > 3.5:
        f1 = p1.y[i] + (3.5 - p1.x[i]) * (( p1.y[i + 1] - p1.y[i]) / (p1.x[i + 1] - p1.x[i]))
        p1.y[i + 1] = f1
        p1.x[i + 1] = 3.5

        speedi = speed(p1.vx, p1.vy, i)
        speedf = speed(p1.vx, p1.vy, i + 1)
        distanceInt = distanceBetw(p1.x, p1.y, i)
        dtInt = (p1.x[i + 1] - p1.x[i]) / p1.vx[i]

        t[i + 1] = t[i] + dtInt
        speedInt = speedf * (dtInt / dt) + speedi * (1 - (dtInt / dt))
        
        p1.vx[i + 1] = -(p1.vx[i + 1] * (dtInt / dt) + p1.vx[i] * (1 - (dtInt / dt)))
        p1.vy[i + 1] = p1.vy[i]
        

        for b in t:
            if(b > i + 1):
                t[b] -= dt - dtInt
        

    if p1.x[i + 1] < 0:
        f1 = p1.y[i] + (0 - p1.x[i]) * ((p1.y[i + 1] - p1.y[i]) / (p1.x[i + 1] - p1.x[i]))
        p1.y[i + 1] = f1
        p1.x[i + 1] = 0

        speedi = speed(p1.vx, p1.vy, i)
        speedf = speed(p1.vx, p1.vy, i + 1)
        distanceInt = distanceBetw(p1.x, p1.y, i)
        dtInt = (p1.x[i + 1] - p1.x[i]) / p1.vx[i]

        t[i + 1] = t[i] + dtInt
        speedInt = speedf * (dtInt / dt) + speedi * (1 - (dtInt / dt))
        
        p1.vx[i + 1] = -(p1.vx[i + 1] * (dtInt / dt) + p1.vx[i] * (1 - (dtInt / dt)))
        p1.vy[i + 1] = p1.vy[i]
        

        for b in t:
            if(b > i + 1):
                t[b] -= dt - dtInt
    
    
    #Collision detection for p2
    if p2.y[i + 1] < 0:
        f2 = p2.x[i] + (0 - p2.y[i]) * ((p2.x[i + 1] - p2.x[i]) /  (p2.y[i + 1] - p2.y[i]))
        p2.y[i + 1] = 0
        p2.x[i + 1] = f2

        speedi = speed(p2.vx, p2.vy, i)
        speedf = speed(p2.vx, p2.vy, i + 1)
        distanceInt = distanceBetw(p2.x, p2.y, i)
        dtInt = (p2.x[i + 1] - p2.x[i]) / p2.vx[i]

        t[i + 1] = t[i] + dtInt

        p2.vx[i + 1] = p2.vx[i]
        p2.vy[i + 1] = -(p2.vy[i + 1] * (dtInt / dt) + p2.vy[i] * (1 - (dtInt / dt))) * collDamp

        for c in range(tIteration):
            if(c > i + 1):
                t[c] -= dt - dtInt

    if p2.x[i + 1] > 3.5:
        f2 = p2.y[i] + (3.5 - p2.x[i]) * (( p2.y[i + 1] - p2.y[i]) / (p2.x[i + 1] - p2.x[i]))
        p2.y[i + 1] = f2
        p2.x[i + 1] = 3.5

        speedi = speed(p2.vx, p2.vy, i)
        speedf = speed(p2.vx, p2.vy, i + 1)
        distanceInt = distanceBetw(p2.x, p2.y, i)
        dtInt = (p2.x[i + 1] - p2.x[i]) / p2.vx[i]

        t[i + 1] = t[i] + dtInt
        speedInt = speedf * (dtInt / dt) + speedi * (1 - (dtInt / dt))
        
        p2.vx[i + 1] = -(p2.vx[i + 1] * (dtInt / dt) + p2.vx[i] * (1 - (dtInt / dt)))
        p2.vy[i + 1] = p2.vy[i]
        

        for b in t:
            if(b > i + 1):
                t[b] -= dt - dtInt
        

    if p2.x[i + 1] < 0:
        f2 = p2.y[i] + (0 - p2.x[i]) * ((p2.y[i + 1] - p2.y[i]) / (p2.x[i + 1] - p2.x[i]))
        p2.y[i + 1] = f2
        p2.x[i + 1] = 0

        speedi = speed(p2.vx, p2.vy, i)
        speedf = speed(p2.vx, p2.vy, i + 1)
        distanceInt = distanceBetw(p2.x, p2.y, i)
        dtInt = (p2.x[i + 1] - p2.x[i]) / p2.vx[i]

        t[i + 1] = t[i] + dtInt
        speedInt = speedf * (dtInt / dt) + speedi * (1 - (dtInt / dt))
        
        p2.vx[i + 1] = -(p2.vx[i + 1] * (dtInt / dt) + p2.vx[i] * (1 - (dtInt / dt)))
        p2.vy[i + 1] = p2.vy[i]
        

        for b in t:
            if(b > i + 1):
                t[b] -= dt - dtInt



scat1 = ax.scatter(p1.x[0], p1.y[0], c="b", s=5**2, label=f'v1 = ({p1.vx[0]}, {p1.vy[0]}) m/s')
scat2 = ax.scatter(p2.x[0], p2.y[0], c="r", s=5, label=f'v2 = ({p2.vx[0]}, {p2.vy[0]}) m/s')

ax.set(xlim=[0, 3.5], ylim=[0, 10], xlabel='X (m)', ylabel='Y (m)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()


def update(frame):
    # for each frame, update the data stored on each artist.
    x1 = p1.x[:frame]
    y1 = p1.y[:frame]
    x2 = p2.x[:frame]
    y2 = p2.y[:frame]
    time_text.set_text('Time = %0.2f' % t[frame])
    # update the scatter plot:
    data1 = np.stack([x1, y1]).T
    scat1.set_offsets(data1)
    data2 = np.stack([x2, y2]).T
    scat2.set_offsets(data2)
    return (scat1, scat2)

plt.scatter
ani = animation.FuncAnimation(fig=fig, func=update, frames=tIteration, interval=1)

plt.show()