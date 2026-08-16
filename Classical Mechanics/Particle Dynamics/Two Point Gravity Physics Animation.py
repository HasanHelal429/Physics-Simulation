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
GravC = 1.5

timeRange = 10
dt = 0.01
tIteration = int(round(timeRange / dt))

fig, ax = plt.subplots()
t = np.linspace(0, timeRange, tIteration)

mass1 = 10
r1 = 0.2
x1 = 5
y1 = 4

v1x = np.sqrt(mass1*GravC/4)
print(v1x)
v1y = 0.01

mass2 = 10
x2 = 5
y2 = 6

v2x = -np.sqrt(mass1*GravC/4)
v2y = 0.01

p1 = pt.Point(x1, y1, v1x, v1y, mass1, r1)
p2 = pt.Point(x2, y2, v2x, v2y, mass2, r1)

speed1 = [speed(p1.vx, p1.vy, 0)]
speed2 = [speed(p2.vx, p2.vy, 0)]

a1x = 0
a1y = 0

a2x = 0
a2y = 0


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

    a1x = GravC * p2.mass * (p2.x[i + 1] - p1.x[i + 1]) / np.sqrt((p2.x[i + 1] - p1.x[i + 1])**2 + (p2.y[i + 1] - p1.y[i + 1])**2)**3
    a1y = GravC * p2.mass * (p2.y[i + 1] - p1.y[i + 1]) / np.sqrt((p2.x[i + 1] - p1.x[i + 1])**2 + (p2.y[i + 1] - p1.y[i + 1])**2)**3

    a2x = GravC * p1.mass * (p1.x[i + 1] - p2.x[i + 1]) / np.sqrt((p2.x[i + 1] - p1.x[i + 1])**2 + (p2.y[i + 1] - p1.y[i + 1])**2)**3
    a2y = GravC * p1.mass * (p1.y[i + 1] - p2.y[i + 1]) / np.sqrt((p2.x[i + 1] - p1.x[i + 1])**2 + (p2.y[i + 1] - p1.y[i + 1])**2)**3
    
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

    if p1.y[i + 1] > 10:
        f1 = p1.x[i] + (10 - p1.y[i]) * ((p1.x[i + 1] - p1.x[i]) /  (p1.y[i + 1] - p1.y[i]))
        p1.y[i + 1] = 10
        p1.x[i + 1] = f1

        speedi = speed(p1.vx, p1.vy, i)
        speedf = speed(p1.vx, p1.vy, i + 1)
        distanceInt = distanceBetw(p1.x, p1.y, i)
        dtInt = -(p1.x[i + 1] - p1.x[i]) / p1.vx[i]

        t[i + 1] = t[i] + dtInt

        p1.vx[i + 1] = p1.vx[i]
        p1.vy[i + 1] = -(p1.vy[i + 1] * (dtInt / dt) + p1.vy[i] * (1 - (dtInt / dt))) * collDamp

        for c in range(tIteration):
            if(c > i + 1):
                t[c] -= dt - dtInt

    if p1.x[i + 1] > 10:
        f1 = p1.y[i] + (10 - p1.x[i]) * (( p1.y[i + 1] - p1.y[i]) / (p1.x[i + 1] - p1.x[i]))
        p1.y[i + 1] = f1
        p1.x[i + 1] = 10

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

    if p2.y[i + 1] > 10:
        f2 = p2.x[i] + (10 - p2.y[i]) * ((p2.x[i + 1] - p2.x[i]) /  (p2.y[i + 1] - p2.y[i]))
        p2.y[i + 1] = 10
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

    if p2.x[i + 1] > 10:
        f2 = p2.y[i] + (10 - p2.x[i]) * (( p2.y[i + 1] - p2.y[i]) / (p2.x[i + 1] - p2.x[i]))
        p2.y[i + 1] = f2
        p2.x[i + 1] = 10

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



scat1 = ax.scatter(p1.x[0], p1.y[0], c="b", s=15, label=f'v1 = ({p1.vx[0]}, {p1.vy[0]}) m/s')
scat2 = ax.scatter(p2.x[0], p2.y[0], c="r", s=15, label=f'v2 = ({p2.vx[0]}, {p2.vy[0]}) m/s')

ax.set(xlim=[0, 10], ylim=[0, 10], xlabel='X (m)', ylabel='Y (m)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()


def update(frame):
    
    time_text.set_text('Time = %0.2f' % t[frame])
    # update the scatter plot:
    scat1.set_offsets((p1.x[frame],p1.y[frame]))
    scat2.set_offsets((p2.x[frame],p2.y[frame]))
    return (scat1, scat2)

plt.scatter
ani = animation.FuncAnimation(fig=fig, func=update, frames=tIteration, interval=1)
ani.save(filename="/Users/hasan/Python Animations/binary_orbit_system.gif", writer="pillow",fps=50)

plt.show()