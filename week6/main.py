import math
import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import sys
import time

minimal = 0
maximum = 0
step = 0

functionName = ""

dimension = 0

mem = []
c1 = 0
c2 = 0

# The bet position of the particles
gBest = np.array(0)
gBestFitness = -1

########################################
######## optimization functions ########
########################################


def testSphereFunction(points):
    sum = 0
    for values in points:
        sum += values**2
    return sum


def testAckleyFunction(points):
    firstSum = 0
    secondSum = 0
    for value in points:
        firstSum += value**2
        secondSum += np.cos(2*math.pi*value)
    d = float(len(points))
    return (-20*np.exp(-0.2*np.sqrt(firstSum/d)) - np.exp(secondSum/d) + 20 + math.e)


def testGriewankFunction(points):
    sum = 0
    prod = 1
    i = 1
    for value in points:
        sum += (value**2)/4000
        prod *= np.cos(value/np.sqrt(i))
        i += 1
    return sum-prod+1


def testRastriginFunction(points):
    d = float(len(points))
    sum = 0
    for value in points:
        sum += ((value**2)-10*np.cos(2*math.pi*value))
    return 10*d + sum


def testSchwefelFunction(points):
    d = float(len(points))
    sum = 0
    for value in points:
        sum += value*np.sin(np.sqrt(np.abs(value)))
    return 418.9829*d - sum


def testMichalewiczFunction(points):
    sum = 0
    i = 1
    for value in points:
        sum += np.sin(value)*np.power(np.sin((i*(value**2))/math.pi), 2*10)
        i += 1
    return -sum


def testRosenbrockFunction(points):
    l = len(points)
    for index, obj in enumerate(points):
        if index == 0:
            sum = 0
        if index < (l - 1):
            sum += 100*(points[index+1] - (obj)**2)**2 + (obj-1)**2
    return sum


def testZakharovFunction(points):
    firstSum = 0
    secondSum = 0
    i = 1
    for value in points:
        firstSum += value**2
        secondSum += 0.5*i*value
        i += 1
    return firstSum + np.power(secondSum, 2) + np.power(secondSum, 4)


def getW(x):
    return 1+(x-1)/4


def testLevyFunction(points):
    sum = 0
    part1 = np.sin(math.pi*getW(points[0]))**2
    wd = getW(points[len(points)-1])
    for index, value in enumerate(points):
        wi = getW(value)
        if index < (len(points)-1):
            sum += ((wi-1)**2) * (1+10*(np.sin(math.pi*wi+1)**2)) + \
                ((wd-1)**2) * (1+np.sin(2*math.pi*wd)**2)
    return part1 + sum


def generateRandomSolution():
    global dimension, minimal, maximum
    point = []
    for i in range(0, dimension):
        point.append(random.uniform(minimal, maximum))
    value = np.asarray(point)
    return value

# Class Particle, each Particle is an object of this class


class Particle:
    """[Class particle that represents one individual of the swarm
            - actualPos : Is the actual position of the particle
            - actualFitness : The fitness of the actual position of the particle
            - pBest : Is the best position of the particle
            - pBestFitness : The fitness of the best position of the particle
            - velocity : Array that changes the direction of the movement of the particle
        ]
    """    
    def __init__(self):
        """Constructor of the class, initializes the properties of the class
        """        
        self.actualPos = generateRandomSolution()
        self.actualFitness = -1
        self.pBest = self.actualPos
        self.pBestFitness = -1
        self.velocity = np.array(0)

    def changePos(self):
        """Function that changes de position of the particle
        based on the velocity vector and
        the actual position
        """        
        global minimal, maximum
        self.changeVel()
        self.actualPos = self.actualPos + self.velocity
        for i in range(len(self.actualPos)):
            if self.actualPos[i] < minimal:
                self.actualPos[i] = minimal
            elif self.actualPos[i] > maximum:
                self.actualPos[i] = maximum
        self.computeParticleFitness()

    def changeVel(self):
        """Function that changes the velocity vector based
        on the best position of the particle and the best
        position of the swarm
        """
        global c1, c2, gBest
        r = random.uniform(0, 1)
        self.velocity = self.velocity + \
            (c1 * r * (self.pBest - self.actualPos) +
             c2 * r * (gBest - self.actualPos))

    def computeParticleFitness(self):
        """Function that computes the fitness of the particle
        and changes the best particle position if it improves it
        
        """        
        self.actualFitness = computeFitness(self.actualPos)
        if (self.actualFitness < self.pBestFitness) or (self.pBestFitness == -1):
            self.pBest = self.actualFitness
            self.pBestFitness = self.actualFitness


def computeFitness(point):
    """Function that computes the fitness of a particle in
    the function
    
    Arguments:
        point {[list]} -- [List of n dimensions with one value for each dimension]
    
    Returns:
        [float] -- [Value of the fitness for that point/solution]
    """    
    global functionName
    fitness = functionName(point)
    return fitness

########################################
################ SOMA ##################
########################################


def initPop(popSize):
    """Function that initializes the population of solutions
    
    Arguments:
        popSize {[int]} -- [Size of the population]
    
    Returns:
        [list] -- [List of solutions (particles)]
    """    
    global minimal, maximum
    pop = []
    for i in range(0, popSize):
        p = Particle()
        pop.append(p)
    print(pop)
    input()
    return pop


def movePop(pop):
    """Function that move each particle to a new position
    
    Arguments:
        pop {[list]} -- [List of particles]
    
    Returns:
        [list] -- [List with the particles moved to new position]
    """    
    global functionName, minimal, maximum, gBestFitness, gBest
    for j in range(len(pop)):
        pop[j].changePos()
        if gBestFitness == -1:
            gBest = pop[j].actualPos
            gBestFitness = pop[j].actualFitness
        else:
            if pop[j].actualFitness < gBestFitness or gBestFitness == -1:
                gBest = pop[j].actualPos
                gBestFitness = pop[j].actualFitness
        print(pop[j].actualPos, pop[j].actualFitness)
    return pop


def runPSExperiment(numGen, popSize, fig):
    """Function that runs the particle swarm experiment
    
    Arguments:
        numGen {[int]} -- [Number of generations]
        popSize {[int]} -- [Size of the population]
        fig {[figure]} -- [Figure to plot the swarm in each movement of population]
    Returns:
        [list] -- [List of solutions at the end of the algorithm]
    """ 
    functionResults = []
    global mem, gBest, gBestFitness, functionName, minimal, maximum
    ax = fig.add_subplot(111)
    # Init the population
    pop = initPop(popSize)
    # Iterate thorugh each generation generating a new population based on the mutations
    # of the previous generation
    for j in range(numGen):
        nextPop = movePop(pop)
        plotSwarm(fig, ax, pop, j)
        pop = nextPop
        mem.append(gBestFitness)
        print("\n\nGEN: ", j, ": ", gBest, gBestFitness, "\n\n")
    functionResults.append([gBestFitness, gBest[0], gBest[1]])
    return functionResults

# For each function run the experiment


def runExperiments(numGen, popSize, fig):
    """Function that runs all the experiments for each function
    
    Arguments:
        numGen {[int]} -- [Number of generations]
        popSize {[int]} -- [Size of the population]
        fig {[figure]} -- [Figure to plot the swarm in each movement of population]
    Returns:
        [list] -- [List of solutions at the end of the algorithm]
    """ 
    results = []
    results.append(runPSExperiment(numGen, popSize, fig))
    return results


def plotSwarm(fig, ax, pop, gen):
    """Function that plots the swarm
    
    Arguments:
        fig {[figure]} -- [Figure to plot the swarm in each movement of population]
        ax {[axis]} -- [Axis of the figure]
        pop {[list]} -- [List of particles]
        gen {[int]} -- [Number of the generation]
    """    
    global minimal, maximum, gBest
    if plt.fignum_exists(1):
        ax.clear()
        title = "gen " + str(gen)
        fig.suptitle(title)
        for j in range(len(pop)):
            ax.plot(pop[j].actualPos[0], pop[j].actualPos[1], markerfacecolor='green',
                    markeredgecolor='blue', marker='o', markersize=5)
        ax.plot(gBest[0], gBest[1], markerfacecolor='blue', markeredgecolor='red', marker='o', markersize=5)
        ax.set_xlim(minimal, maximum)
        ax.set_ylim(minimal, maximum)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.2)


def plotMem():
    """Function that plots the memory created through the generations
    """    
    global mem
    fig, ax = plt.subplots()
    ax.plot(mem)

    ax.set(xlabel='Gens', ylabel='Fitness')
    ax.grid()

def plotFunction(fig):
    """Plot the optimization function
    
    Arguments:
        fig {[figure]} -- [Figure to plot the swarm in each movement of population]    
    
    Returns:
        [axis] -- [Returns the axis for that figure]
    """    
    global minimal, maximum, functionName, step
    ax = plt.axes(projection="3d")

    X = np.arange(minimal, maximum, step)
    Y = np.arange(minimal, maximum, step)
    X, Y = np.meshgrid(X, Y)
    Z = []
    z_rows = []
    for xi, y_i in zip(X, Y):
        point = [xi, y_i]
        z_rows.append(functionName(point))
    Z = np.array(z_rows)
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',
                           linewidth=0, antialiased=False, alpha=0.35)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return ax


def main():
    global c1, c2, dimension, minimal, maximum, functionName, step
    if len(sys.argv) != 7:
        print("Incorrect number of parameters: \n" +
              "\t main.py \"Function Name\" \"dimension\" \"nGens\" \"popSize\" \"c1\" \"c2\"")
        return -1
    # Dictionary and arrays of the values for each function
    functionMin = [-6, -33, -600, -6, -500, 0, -10, -10, -10]
    functionMax = [6, 33, 600, 6, 500, 4, 10, 10, 10]
    functionStep = [0.01, 0.25, 1, 0.01, 1, 0.01, 0.25, 0.25, 0.25]
    dispatcher = {'testSphereFunction': testSphereFunction,
                  'testAckleyFunction': testAckleyFunction,
                  'testGriewankFunction': testGriewankFunction,
                  'testRastriginFunction': testRastriginFunction,
                  'testSchwefelFunction': testSchwefelFunction,
                  'testMichalewiczFunction': testMichalewiczFunction,
                  'testRosenbrockFunction': testRosenbrockFunction,
                  'testZakharovFunction': testZakharovFunction,
                  'testLevyFunction': testLevyFunction}
    functionArgv = sys.argv[1]
    dimension = int(sys.argv[2])
    nGens = int(sys.argv[3])
    popSize = int(sys.argv[4])
    c1 = float(sys.argv[5])
    c2 = float(sys.argv[6])
    # Parse the arg to function
    functionName = dispatcher[functionArgv]
    # Get the index of the function, to get the minimal, maximum and step for plotting
    index = list(dispatcher.keys()).index(functionArgv)
    minimal = functionMin[index]
    maximum = functionMax[index]
    step = functionStep[index]
    plt.ion()
    fig = plt.figure(1)
    obtainedSolutions = runExperiments(nGens, popSize, fig)
    plt.ioff()
    abcises = []
    # Append the plot for the function
    fig = plt.figure(2)
    abcises.append(plotFunction(fig))

    # Plot the best solution for that optimization function
    for functionSols, ax in zip(obtainedSolutions, abcises):
        x_sol = []
        y_sol = []
        z_sol = []
        for point in functionSols:
            z_sol.append(point[0])
            x_sol.append(point[1])
            y_sol.append(point[2])

        ax.plot(x_sol, y_sol, z_sol, markerfacecolor='green',
                markeredgecolor='blue', marker='o', markersize=10)
    plotMem()
    plt.show()


if __name__ == "__main__":
    main()
