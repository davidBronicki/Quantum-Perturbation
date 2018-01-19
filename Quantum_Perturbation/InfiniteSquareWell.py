from numpy import *
from Quantum_Perturbation import QuantumPerturbation as qp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

length = 1
mass = 1
hbar = 1

def eigenEnergy(n):
    return (n*pi*hbar/length)**2/(2*mass)

def eigenFunction(n):
    const = sqrt(2)
    const2 = n*pi/length
    return lambda x: const * sin(const2*x)

def oneDim():
    def graph(*args):
        n = 1000
        dx = 1 / n
        input = arange(dx / 2, 1, dx)
        for function in args:
            output = []
            for x in input:
                output.append(function(x))
            plt.plot(input, output)
        plt.show()

    bounds = [(0, 1)]
    size = 50
    energyStuff =[]
    for i in range(1, size+1):
        energyStuff.append([eigenEnergy(i), eigenFunction(i)])

    sys = qp.QuantumSystem(energyStuff, bounds, lambda x: 1, 'function')

    omega = 80
    divs = 50
    perturbationFunction = lambda x: .5*mass*omega**2*(x-.5)**2/divs
    plt.plot(sys.energyLevels)
    for i in range(divs):
        sys = sys.perturb(perturbationFunction)
        if i % 10 == 0:
            sys.normalize()
    print(sys.energyLevels)
    plt.plot(sys.energyLevels)
    plt.show()

    for n in [0,1,4,7,8,20]:
        graph(sys.getState(n), eigenFunction(n+1))

def twoDim():
    def generate2DFunction(n, m):
        f1 = eigenFunction(n)
        f2 = eigenFunction(m)
        return lambda x, y: f1(x) * f2(y)

    def graph(*args):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        n = 20
        dx = 1/n
        inputX = arange(dx/2, 1, dx)
        inputY = inputX.copy()
        outputX, outputY = meshgrid(inputX, inputY)
        i = 0
        for function in args:
            outputZ = []
            for x in inputX:
                temp = []
                for y in inputY:
                    temp.append(function(x, y))
                outputZ.append(temp)
            outputZ = array(outputZ)
            if i == 0:
                ax.plot_wireframe(outputX, outputY, outputZ)
            else:
                ax.plot_wireframe(outputX, outputY, outputZ, color = 'red')
            i += 1
        plt.show()

    bounds = [(0,1),(0,1)]
    size = 20
    energyStuff = []
    energyGraphsOriginal = []
    for n in range(1, size + 1):
        max = int(sqrt(21**2-n**2)+.5)
        for m in range(1, max):
            energyGraphsOriginal.append([eigenEnergy(n) + eigenEnergy(m), generate2DFunction(n, m)])
            energyStuff.append(energyGraphsOriginal[-1])
    print(len(energyGraphsOriginal))
    throwout,energyGraphsOriginal = qp.formatEnergyStuff(energyGraphsOriginal)
    energyGraphsOriginal = list(zip(*energyGraphsOriginal))[1]

    sys = qp.QuantumSystem(energyStuff, bounds, lambda x, y: 1, 'function')

    omega = 40
    divs = 10
    perturbationFunction = lambda x, y: .5 *mass* omega**2*((x-.5)**2 + (y-.5)**2)/divs
    plt.plot(sys.energyLevels)
    for i in range(divs):
        t0 = time()
        sys = sys.perturb(perturbationFunction)
        sys.normalize()
        # if i % 2 == 0:
        #     sys.normalize()
        print('completed a step in ', time() - t0, ' seconds')
    print(sys.energyLevels)
    plt.plot(sys.energyLevels)
    plt.show()

    for n in [0,1,2,3,4,5,6,7,8,9]:
        graph(sys.getState(n), energyGraphsOriginal[n])

# oneDim()
twoDim()
