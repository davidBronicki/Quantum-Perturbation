import Quantum_Perturbation.PsuedoFunctionLattice as lat
from numpy import *

def formatEnergyStuff(inputStuff):
    inputStuff.sort(key = lambda x: x[0])
    sizes = []
    for i in range(len(inputStuff)):
        try:
            temp = inputStuff[i][0]/inputStuff[i-1][0]
            if temp < .99 or temp > 1.01:
                sizes.append(1)
            else:
                sizes[-1] += 1
        except:
            sizes.append(1)
    return sizes, inputStuff

class QuantumSystem:

    def __init__(self, energyStuff, bounds, differentialStuff, energyInfo = None, differentialInfo = None):
        self.sizeGuide, temp = formatEnergyStuff(energyStuff)
        self.energyLevels, temp = list(zip(*temp))
        self.energyLevels = array(self.energyLevels)
        if energyInfo == 'lattice':
            self.states = temp
        else:#function
            self.states = []
            for func in temp:
                self.states.append(lat.constructLattice(func, bounds))
        self.size = sum(self.sizeGuide)
        self.bounds = bounds
        if differentialInfo == 'lattice':
            self.differentialLattice = differentialStuff
        else:#function
            self.differentialLattice = lat.constructLattice(differentialStuff, bounds) * lat.constructDifferentialLattice(bounds)

    def perturb(self, perturbationFunction):
        perturbationLattice = lat.constructLattice(perturbationFunction, self.bounds)
        k = 0
        nonDegenerateLats = []
        newEnergies = []
        for item in self.sizeGuide:
            tempMatrix = zeros((item, item))
            for i in range(item):
                for j in range(i,item):
                    temp = perturbationLattice * self.states[i + k] * self.states[j + k].conjugate() * self.differentialLattice
                    temp = temp.integrate()
                    tempMatrix[i][j] = temp
                    tempMatrix[j][i] = temp.conjugate()
            eigenValues, eigenVectors = linalg.eigh(tempMatrix)
            newEnergies.extend(eigenValues)
            for vec in eigenVectors:
                newThing = self.states[k:k + item]
                newLat = lat.constructLattice(lambda *args: 0, self.bounds)
                for i in range(item):
                    newLat += newThing[i] * vec[i]
                nonDegenerateLats.append(newLat)
            k += item
        k=0
        newLats = []
        for item in self.sizeGuide:
            subspace = range(k, k+item)
            for i in subspace:
                newLat = lat.constructLattice(lambda *args: 0, self.bounds)
                for j in range(self.size):
                    if j not in subspace:
                        temp = nonDegenerateLats[i] * nonDegenerateLats[j].conjugate() * perturbationLattice * self.differentialLattice
                        temp = temp.integrate() / (self.energyLevels[i] - self.energyLevels[j])
                        newLat += temp * nonDegenerateLats[j]
                newLats.append(newLat)
            k += item
        for i in range(self.size):
            newLats[i] += self.states[i]
            newEnergies[i] += self.energyLevels[i]
        energyStuff = list(zip(newEnergies, newLats))
        return QuantumSystem(energyStuff, self.bounds, self.differentialLattice, 'lattice', 'lattice')

    def normalize(self):
        newStates = []
        for thing in self.states:
            temp = thing*thing.conjugate()*self.differentialLattice
            temp = temp.integrate()
            newStates.append(thing / sqrt(temp))
        self.states = newStates

    def getState(self, n):
        return self.states[n].interpolationFunction()
