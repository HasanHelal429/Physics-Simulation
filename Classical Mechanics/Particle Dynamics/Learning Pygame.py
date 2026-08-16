# Simple pygame program

# Import the pygame library
import pygame
import numpy as np
import random as rand

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

class Point:
    def __init__(self, x, y, vx, vy, mass, radius):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.radius = radius
    
    def getX(self, i):
        return self.x[i]

    def getY(self, i):
        return self.y[i]
    
    def speed(self):
        return np.sqrt(self.vx**2 + self.vy**2)

# This function calculates the Distance between any two points
def distanceBetwPts(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def boundMax(value, max):
    if(value > max):
        return max
    else:
        return value

def ParticleColl(p, nextX, nextY, nextVX, nextVY, i, numP):
     # Collision detection between particles
    for q in range(11):
        k = j - 5 + q
        if(k >= 0 and k < numP):
            if(distanceBetwPts(p[i], p[k]) < (p[i].radius + p[k].radius) and k != i):
                collHap[i] = True
                nextVX[i] = p[i].vx - 2*p[k].mass*(p[i].x - p[k].x)*((p[i].vx - p[k].vx)*(p[i].x - p[k].x)
                                    + (p[i].vy - p[k].vy)*(p[i].y - p[k].y))/((p[i].mass + p[k].mass)*
                                                                                            distanceBetwPts(p[i], p[k])**2)
                nextVY[i] = p[i].vy - 2*p[k].mass*(p[i].y - p[k].y)*((p[i].vx - p[k].vx)*(p[i].x - p[k].x)
                                    + (p[i].vy - p[k].vy)*(p[i].y - p[k].y))/((p[i].mass + p[k].mass)*
                                                                                                distanceBetwPts(p[i], p[k])**2)
                nextX[i] = p[i].x + dt * nextVX[i] + dt**2 * ax1 / 2
                nextY[i] = p[i].y + dt * nextVY[i] + dt**2 * ay1 / 2

def wallColl(p1, i, lWall, rWall, tWall, bWall):
    if(p1[i].y < tWall + r1):
        nextVY[i] *= -1
        nextY[i] = tWall + r1
    if(p1[i].y > bWall - r1):
        nextVY[i] *= -0.8
        nextY[i] = bWall - r1

    if(p1[i].x > rWall - r1):
        nextVX[i] *= -1
        nextX[i] = rWall - r1
    if(p1[i].x < lWall + r1):
        nextVX[i] *= -1
        nextX[i] = lWall + r1

def scmInsertionSort(p1,scm1,nextx,nexty,nextvx,nextvy):
    n = len(scm1)  # Get the length of the array
     
    if n <= 1:
        return  # If the array has 0 or 1 element, it is already sorted, so return

    for i in range(1, n):  # Iterate over the array starting from the second element
        key1 = scm1[i]  # Store the current element as the key to be inserted in the right position
        key2 = p1[i]
        key3 = nextx[i]
        key4 = nexty[i]
        key5 = nextvx[i]
        key6 = nextvy[i]
        j = i-1
        while j >= 0 and key1 < scm1[j]:  # Move elements greater than key one position ahead
            scm1[j+1] = scm1[j]  # Shift elements to the right
            p1[j+1] = p1[j]
            nextx[j+1]=nextx[j]
            nexty[j+1]=nexty[j]
            nextvx[j+1]=nextvx[j]
            nextvy[j+1]=nextvy[j]
            j -= 1
        scm1[j+1] = key1  # Insert the key in the correct position   
        p1[j+1]=key2
        nextx[j+1]=key3
        nexty[j+1]=key4
        nextvx[j+1]=key5
        nextvy[j+1]=key6


# Initialize pygame
pygame.init()

pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
my_font = pygame.font.SysFont('Times New Roman', 30)

clock = pygame.time.Clock()

# Define constants for the screen width and height
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 700

# Define bounds of system
leftWall = 0
rightWall = 1500
ceiling = 0
floor = 700

# Set up the drawing window
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

numP = 20

mass1 = 1
r1 = 6

# Position and velocity values of ball
x1 = 750
y1 = 350

vx1 = 10
vy1 = 0

ax1 = 0
ay1 = 1

p1 = [Point(x1, y1, vx1, vy1, mass1, r1)]
for j in range(numP - 1):
    p1.append(Point(rand.random() * 1500, rand.random() * 700, -vx1, vy1, mass1, r1))

# Create spacial comparison metric
scm = []
for i in range(numP):
    scm.append(p1[i].x * p1[i].y)

nextX = [p1[0].x]
nextY = [p1[0].y]
nextVX = [p1[0].vx]
nextVY = [p1[0].vy]
collHap = [False]
for j in range(numP - 1):
    nextX.append(p1[j + 1].x)
    nextY.append(p1[j + 1].y)
    nextVX.append(p1[j + 1].vx)
    nextVY.append(p1[j + 1].vy)
    collHap.append(False)

dt = 1/60

# Run until the user asks to quit
running = True
while running:

    clock.tick()
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            # Was it the Escape key? If so, stop the loop.
            if event.key == K_ESCAPE:
                running = False
            
            if event.key == K_LEFT:
                nextVX[0] += -10
            
            if event.key == K_RIGHT:
                nextVX[0] += 10

            if event.key == K_UP:
                nextVY[0] += -10
            
            if event.key == K_DOWN:
                nextVY[0] += 10
        # Did the user click the window close button? If so, stop the loop.
        elif event.type == QUIT:
            running = False

    for i in range(numP):
        p1[i].x = nextX[i]
        p1[i].y = nextY[i]
        p1[i].vx = nextVX[i]
        p1[i].vy = nextVY[i]

    

    for j in range(numP):
        scm[j]= p1[j].x * p1[j].y
    #print(p1[0].x,p1[0].y,",", p1[1].x,p1[1].y, scm)
    scmInsertionSort(p1,scm,nextX,nextY,nextVX,nextVY)
    #print(p1[0].x,p1[0].y,",", p1[1].x,p1[1].y)
    
    for i in range(numP):
        nextX[i] += p1[i].vx * dt + (ax1 * dt**2 / 2)
        nextY[i] += p1[i].vy * dt + (ay1 * dt**2 / 2)
        
        nextVX[i] += ax1 * dt
        nextVY[i] += ay1 * dt
        
        collHap[i] = False
        # Collision detection between particles
        ParticleColl(p1, nextX, nextY, nextVX, nextVY, i, numP )

        # Collision with wall
        wallColl(p1, i, leftWall, rightWall, ceiling, floor)
        

    # Fill the background with Black
    screen.fill((0, 0, 0))

    for i in range(numP):
        # Draw a solid blue circle in the center
        if(i == 0):
            pygame.draw.circle(screen, (0, 255, 0), (p1[i].x, p1[i].y), p1[i].radius)
        else:
            pygame.draw.circle(screen, (boundMax(p1[i].speed() * 5, 255), 5, 255 - boundMax(p1[i].speed() * 5, 255)), (p1[i].x, p1[i].y), p1[i].radius)
    
    text_surface = my_font.render(str(int(clock.get_fps())), False, (255, 255, 255))
    screen.blit(text_surface, (0,0))

    # Flip the display
    pygame.display.flip()
    pygame.time.get_ticks()
# Done! Time to quit.
pygame.quit()