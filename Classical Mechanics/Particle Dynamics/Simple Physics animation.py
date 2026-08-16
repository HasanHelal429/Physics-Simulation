import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

fig, ax = plt.subplots()
 
timeRange = 7.5
dt = 0.02
tIteration = 50

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

Func = [np.cos(x)]

for i in range(tIteration):
    Func.append(np.cos(x - i/50))


def animate(i):
    line.set_ydata(Func[i])  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50, frames=tIteration,)

plt.show()