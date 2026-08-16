import Point as pt

rowsO, colsO = (9, 2)

# method 2 2nd approach
offsets = [[0 for i in range(colsO)] for j in range(rowsO)]

offsets[0] = [-1,1]
offsets[1] = [0, 1]
offsets[2] = [1, 1]
offsets[3] = [-1, 0]
offsets[4] = [0, 0]
offsets[5] = [1, 0]
offsets[6] = [-1, -1]
offsets[7] = [0, -1]
offsets[8] = [1, -1]


hashK1 = 15823
hashK2 = 9737333

def GetCell(p1, cellSize):
    return (p1.x / cellSize), (p1.y / cellSize)

def HashCell(cellX, cellY):
    return (cellX * hashK1 + cellY*hashK2)

def KeyFromHash(hash, tableSize):
	return hash % tableSize