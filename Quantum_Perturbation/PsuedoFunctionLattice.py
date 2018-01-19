'''This file contains the Lattice class and several helper functions.
The lattice class is meant to act like a function for the purposes of
integration. It consists of a lattice of points at which the parent function
had already been evaluated. In this way, integration becomes simply multipying
lattices together and summing up the indices. This gives a performance boost
compared to the technique of evaluating a true function every time.

A result of this teqnique is that a new tactic must be used to evaluate infinite
integrals. Here, three substitutions are used, and the appropriate differential
terms are created for each.
For (-inf, inf),    y = 2 / (1 + e**(sigma(x-mu)))-1.
For (-inf, b),      y = 1 / (1 + e**(sigma(x-a))).
For (a, inf),       y = 1 / (1 + e**(-sigma(x-a))).'''

from numpy import *

#pointCount must be 4 or more
pointCount = 100

#format of bounds is (lower, upper, sigma, mu)

def prod(inputList):
    #multiply all of the elements of a list together
    output = 1
    for element in inputList:
        output *= element
    return output

def constructDifferentialLattice(bounds):
    '''This corresponds to the differential term of an integral.
    For example, r**2 * sin(theta) for spherical coordinates. Due to the way
    infinte bounds are handled, some strange substitutions are made.
    See explanation at top.'''
    base = arange(1/(2 * pointCount), 1, 1 / pointCount)
    vals = []
    for bound in bounds:
        val = findSwitchNumber(bound)
        if val == 0:#(a, b)
            a = bound[0]
            b = bound[1]
            vals.append(pointCount * [b - a])
        elif val == 1:#(a, inf)
            a = bound[0]
            sigma = 1/bound[2]
            vals.append(1/(sigma*base))
        elif val == 2:#(-inf, b)
            b = bound[1]
            sigma = 1/bound[2]
            vals.append(1/(sigma*base))
        else:#(-inf, inf)
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
    #finds the "switch number". For use in pseudo switch blocks.
    if bound[0] == -inf:
        if bound[1] == inf:#(-inf, inf)
            return 3
        else:#(-inf, b)
            return 2
    elif bound[1] == inf:#(a, inf)
        return 1
    else:#(a, b)
        return 0

def constructSamplePoints(bounds):
    #Construct the set of points to be sampled for the lattice. Due to the way
    #infinte bounds are handled, the point distribution can be quite strange.
    #See explanation at top.
    base = arange(1/(2 * pointCount), 1, 1 / pointCount)
    meshList = []
    for bound in bounds:
        val = findSwitchNumber(bound)
        if val == 0:#(a, b)
            a = bound[0]
            b = bound[1]
            meshList.append(base*(b-a)+a)
        elif val == 1:#(a, inf)
            b = bound[1]
            sigma = 1 / bound[2]
            meshList.append(log(base) / sigma + b)
        elif val == 2:#(-inf, b)
            a = bound[0]
            sigma = 1 / bound[2]
            meshList.append(a - log(base) / sigma)
        else:#(-inf, inf)
            sigma = 1/bound[2]
            mu = bound[3]
            meshList.append(log(1/base - 1)/sigma + mu)
    return meshList

def meshifyPoints(*args):
    #take n lists and return (listSize)**n points
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
    testPoints = constructSamplePoints(bounds)#[x positions, y positions, etc]
    testPoints = meshifyPoints(*testPoints)#[p1, p2, p3, etc]
    vals = []
    for point in testPoints:
        vals.append(function(*point))
    vals = array(vals)
    lattice = reshape(vals, len(bounds) * [pointCount])
    return Lattice(lattice, vals, bounds)

def splineInterpolationFunction(inputY, bounds):
    #cubic interpolation and extrapolation function for n dimensional input list/function
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
        if type(other) != Lattice:
            return NotImplemented
        newLat = self.lattice + other.lattice
        newFlat = self.flatLattice + other.flatLattice
        return Lattice(newLat, newFlat, self.bounds)

    def __mul__(self, other):
        if type(other) != Lattice:
            return Lattice(self.lattice * other, self.flatLattice * other, self.bounds)
        else:
            try:
                return Lattice(self.lattice * other.lattice, self.flatLattice * other.flatLattice, self.bounds)
            except:
                return NotImplemented

    def __truediv__(self, other):
        if type(other) != Lattice:
            return Lattice(self.lattice / other, self.flatLattice / other, self.bounds)
        else:
            try:
                return Lattice(self.lattice / other.lattice, self.flatLattice / other.flatLattice, self.bounds)
            except:
                return NotImplemented

    def __rmul__(self, other):
        Try:
            return self * other
        Except:
            Raise(TypeError)

    def __str__(self):
        return str(self.lattice)

    def integrate(self):
        #Should only be done once differential lattice has been multiplied in.
        return sum(self.flatLattice)

    def interpolationFunction(self):
        return splineInterpolationFunction(self.lattice, self.bounds)

    def conjugate(self):
        return Lattice(self.lattice.conjugate(), self.flatLattice.conjugate(), self.bounds)
