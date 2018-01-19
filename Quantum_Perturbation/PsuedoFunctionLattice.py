from numpy import *

#pointCount must be 4 or more
pointCount = 100

#(a, b, sigma, mu)

def prod(inputList):
    output = 1
    for element in inputList:
        output *= element
    return output

def constructDifferentialLattice(bounds):
    base = arange(1/(2 * pointCount), 1, 1 / pointCount)
    vals = []
    for bound in bounds:
        val = findSwitchNumber(bound)
        if val == 0:
            a = bound[0]
            b = bound[1]
            vals.append(pointCount * [b - a])
        elif val == 1:
            a = bound[0]
            sigma = 1/bound[2]
            vals.append(1/(sigma*base))
        elif val == 2:
            b = bound[1]
            sigma = 1/bound[2]
            vals.append(1/(sigma*base))
        else:
            sigma = 1/bound[2]
            mu = bound[3]
            vals.append(1/(sigma*(base-base**2)))
    vals = meshifyPoints(*vals)
    for i in range(len(vals)):
        vals[i] = prod(vals[i])
    vals = array(vals)
    vals = vals/ pointCount**len(bounds)
    lattice = reshape(vals, len(bounds) * [pointCount])
    return Lattice(lattice, vals, bounds)

def findSwitchNumber(bound):
    if bound[0] == -inf:
        if bound[1] == inf:
            return 3
        else:
            return 2
    elif bound[1] == inf:
        return 1
    else:
        return 0

def constructSamplePoints(bounds):
    base = arange(1/(2 * pointCount), 1, 1 / pointCount)
    meshList = []
    for bound in bounds:
        val = findSwitchNumber(bound)
        if val == 0:
            a = bound[0]
            b = bound[1]
            meshList.append(base*(b-a)+a)
        elif val == 1:
            b = bound[1]
            sigma = 1 / bound[2]
            meshList.append(log(base) / sigma + b)
        elif val == 2:
            a = bound[0]
            sigma = 1 / bound[2]
            meshList.append(a - log(base) / sigma)
        else:
            sigma = 1/bound[2]
            mu = bound[3]
            meshList.append(log(1/base - 1)/sigma + mu)
    return meshList

def meshifyPoints(*args):
    outputMesh = []
    for i in range(len(args)):
        outputMesh.append([])
        outputMesh[i] = args[i]
        for j in range(len(args) - i - 1):
            temp = []
            for element in outputMesh[i]:
                for k in range(len(args[0])):
                    temp.extend([element])
            outputMesh[i] = temp
        for j in range(i):
            outputMesh[i] = len(args[0]) * list(outputMesh[i])
    return list(zip(*outputMesh))

def constructLattice(function, bounds):
    testPoints = constructSamplePoints(bounds)
    testPoints = meshifyPoints(*testPoints)
    vals = []
    for point in testPoints:
        vals.append(function(*point))
    vals = array(vals)
    lattice = reshape(vals, len(bounds) * [pointCount])
    return Lattice(lattice, vals, bounds)

def splineInterpolationFunction(inputY, bounds):
    if len(bounds) == 0:
        return lambda : inputY
    xVals = (constructSamplePoints(bounds))[0]
    interpFunctions = []
    for item in inputY:
        interpFunctions.append(splineInterpolationFunction(item, bounds[1:]))
    def outputFunction(*args):
        x = args[0]
        otherArgs = args[1:]
        i = 0
        sign = xVals[-1] - xVals[0]
        sign /= abs(sign)
        for val in xVals:
            if x*sign < val*sign: break
            else: i+=1
        i-=1
        if i == 0:
            i = 1
        if i == len(xVals) - 2:
            i -= 1
        if i == -1:
            dx = xVals[2]-xVals[0]
            x1 = xVals[0] - dx
            y2 = interpFunctions[0](*otherArgs)
            temp = interpFunctions[2](*otherArgs)
            dy = temp - y2
            k1 = dy/dx
            k2 = k1
            y1 = y2 - dy
        elif i == len(xVals) - 1:
            dx = xVals[-1]-xVals[-3]
            x1 = xVals[-1]
            y1 = interpFunctions[-1](*otherArgs)
            temp = interpFunctions[-3](*otherArgs)
            dy = y1 - temp
            k1 = dy/dx
            k2 = k1
            y2 = y1 + dy
        else:
            x1 = xVals[i]
            x2 = xVals[i+1]
            dx = x2 - x1
            y1 = interpFunctions[i](*otherArgs)
            y2 = interpFunctions[i+1](*otherArgs)
            dy = y2 - y1
            k1 = (y2 - interpFunctions[i-1](*otherArgs))/(x2-xVals[i-1])
            k2 = (interpFunctions[i+2](*otherArgs) - y1)/(xVals[i+2]-x1)
        t = (x-x1)/dx
        a = k1*dx - dy
        b = -k2*dx + dy
        return (1-t)*y1 + t*y2 + t*(1-t)*(a*(1-t)+b*t)
    return outputFunction

class Lattice:
    def __init__(self, lattice, flatLattice, bounds):
        self.lattice = lattice
        self.flatLattice = flatLattice
        self.bounds = bounds

    def __add__(self, other):
        newLat = self.lattice + other.lattice
        newFlat = self.flatLattice + other.flatLattice
        return Lattice(newLat, newFlat, self.bounds)

    def __mul__(self, other):
        if type(other) != Lattice:
            return Lattice(self.lattice * other, self.flatLattice * other, self.bounds)
        else:
            return Lattice(self.lattice * other.lattice, self.flatLattice * other.flatLattice, self.bounds)

    def __truediv__(self, other):
        if type(other) != Lattice:
            return Lattice(self.lattice / other, self.flatLattice / other, self.bounds)
        else:
            return Lattice(self.lattice / other.lattice, self.flatLattice / other.flatLattice, self.bounds)

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return str(self.lattice)

    def integrate(self):
        return sum(self.flatLattice)

    def interpolationFunction(self):
        return splineInterpolationFunction(self.lattice, self.bounds)

    def conjugate(self):
        return Lattice(self.lattice.conjugate(), self.flatLattice.conjugate(), self.bounds)
