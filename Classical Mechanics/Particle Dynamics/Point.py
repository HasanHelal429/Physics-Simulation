import numpy as np

class Point:
    def __init__(self, x, y, vx, vy, mass, radius):
        self.x = [x]
        self.y = [y]
        self.vx = [vx]
        self.vy = [vy]
        self.mass = mass
        self.radius = radius
    
    def getX(self, i):
        return self.x[i]

    def getY(self, i):
        return self.y[i]
    
    def speed(self, i):
        return np.sqrt(self.vx[i]**2 + self.vy[i]**2)
