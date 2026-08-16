import numpy as np
from matplotlib import pyplot as plt, animation

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
ax.set(xlim=(-3, 3), ylim=(-1, 1))
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

x = np.linspace(-3, 3, 150)
t = np.linspace(0, 10,60)
X2, T2 = np.meshgrid(x, t)

sinT2 = np.sin(2 * np.pi * T2 / T2.max()) # How the function evolves in time
F = 0.9 * np.sin(X2 - sinT2) # How the function evolves

line, = ax.plot(x, F[0, :], color='b', lw=1)
def animate(i):
   line.set_ydata(F[i, :])
   time_text.set_text('Time = %0.2f' % t[i])
anim = animation.FuncAnimation(fig, animate, interval=1, frames=len(t) - 1)
#anim.save('503.gif')
plt.show()