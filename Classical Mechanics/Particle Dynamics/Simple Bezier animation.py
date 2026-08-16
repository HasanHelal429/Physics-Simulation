import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

timeRange = 1
dt = 0.02
tIteration = int(round(timeRange / dt))

fig, ax = plt.subplots()
t = np.linspace(0, timeRange, tIteration)

xi = 1
yi = 1

xCntrl = 2
yCntrl = 3

xf = 5
yf = 3

x = ((1 - t)**2) * xi + 2 * (1 - t) * t * xCntrl + t**2 * xf
y = ((1 - t)**2) * yi + 2 * (1 - t) * t * yCntrl + t**2 * yf

bezier =  ax.plot(xi, yi, label=f'v0 =  m/s')[0]
ax.set(xlim=[0, 10], ylim=[0, 10], xlabel='X (m)', ylabel='Y (m)')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()


def update(frame):
    # for each frame, update the data stored on each artist.
    time_text.set_text('Time = %0.2f' % t[frame])
    bezier.set_xdata(x[:frame])
    bezier.set_ydata(y[:frame])
    return (bezier)

plt.scatter
ani = animation.FuncAnimation(fig=fig, func=update, frames=tIteration, interval=5)

plt.show()