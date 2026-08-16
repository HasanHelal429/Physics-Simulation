import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

#Create a function to print the position or velocity
def PosVelprint(x, y):
    print("(", x, ",", y, ")")

#define constants
col_Loss = 0.9

#Define the time and the time steps
t = 0
dt0 = 0.4

#Define the position, velocity, and accerleration of a particle
xPos1, yPos1 = 0, 15
xVel1, yVel1 = 2, 1
xAcc1, yAcc1 = 0, -9.8

speed1 = np.sqrt( xVel1**2 + yVel1**2)

plt.title("Movement")
plt.xlabel("X")
plt.ylabel("Y")
groundx = np.array([0, 10, 10])
groundy = np.array([0, 0, 15])
    
plt.plot(groundx, groundy)
plt.scatter(xPos1, yPos1, marker = 'o', c = int(round(speed1 + 0.5)), cmap = 'viridis')
plt.ion()
plt.show()

while t <= 20: #A for loop that iterates through time
    speed1 = np.sqrt( xVel1**2 + yVel1**2)
    #print(speed1)
    
    #scale the value of dt to account for velocity
    if(speed1 == 0):
        dt = dt0
    else:
        dt = min(dt0 / speed1, dt0)
    

    #iterate the position based on velocity
    xPos1 += dt * xVel1
    yPos1 += dt * yVel1 + dt**2 * yAcc1 / 2

    #iterate the velocity based on position
    xVel1 += dt * xAcc1
    yVel1 += dt * yAcc1

    #Create a ground that bounces the object
    if yPos1 < 0:
        yVel1 *= -1 * col_Loss

    #Create an obstacle that bounces the object
    if xPos1 > 10:
        xVel1 *= -1 * col_Loss
    
    #PosVelprint(xPos1, yPos1)
    

    plt.scatter(xPos1, yPos1, marker = 'o', c = int(round(speed1 + 0.5)), cmap = 'viridis')
    plt.pause(0.01)
    
    t += dt
    print(t)

    time.sleep(0.01)
plt.colorbar()
plt.show()