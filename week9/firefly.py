import math
import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import sys
import time
import click
from scipy.spatial import distance


minimal = 0
maximum = 0
step = 0

functionName = ""

dimension = 0

mem = []
beta_zero = 0
gamma = 0

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

def eudis(v1, v2):
    d = distance.euclidean(v1, v2)
    if d == 0:
        d = 0.01
    return d

class Firefly:
    """Class Firefly, each Firefly is an object of this class
    """    
    def __init__(self):
        """Constructor of the class Firefly
        """        
        self.actualPos = generateRandomSolution()
        self.actualFitness = -1
        self.pBest = self.actualPos
        self.pBestFitness = -1
        self.velocity = np.array(0)

    def changePos(self, pop, index):
        """Function that changes the position of the firefly
        
        Arguments:
            pop {[list]} -- [List of fireflies]
            index {[int]} -- [Index of the firefly]
        """        
        global minimal, maximum, gBest, beta_zero
        lighIntensity = self.getLightIntensity(pop[0])
        mostIntensIndex = 0
        for i in range(index, len(pop)):
            if i != index:
                actualLight = self.getLightIntensity(pop[i])
                if actualLight < lighIntensity:
                    lighIntensity = actualLight
                    mostIntensIndex = i
            r = np.random.uniform()
            distance = eudis(pop[mostIntensIndex].actualPos, self.actualPos)
            newpos = self.actualPos + beta_zero * \
                (1/distance)* \
                (pop[mostIntensIndex].actualPos-self.actualPos) + \
                0.3*(r - 0.5)
            newfitness = computeFitness(newpos)  
            self.actualPos = newpos
            self.actualFitness = newfitness
            
        for i in range(len(self.actualPos)):
            if self.actualPos[i] < minimal:
                self.actualPos[i] = minimal
            elif self.actualPos[i] > maximum:
                self.actualPos[i] = maximum
        self.computeParticleFitness()

    def getLightIntensity(self, p):
        """Function that gets the ligth intensity (Fitness) of a firefly
        
        Arguments:
            p {[Firefly]} -- [Firefly object]
        
        Returns:
            [float] -- [Fitness for that firefly]
        """        
        I = computeFitness(p.actualPos)
        return I

    def getMovement(self, p):
        """Function that calculates the beta value, which is needed for the movement of the firefly
        
        Arguments:
            p {[Firefly]} -- [Firefly object]
        
        Returns:
            [float] -- [Value of the beta]
        """        
        global beta_zero, gamma
        beta = beta_zero * \
            np.exp((-gamma)*(np.linalg.norm(p.actualPos-self.actualPos)))
        return beta

    def computeParticleFitness(self):
        """Function that computes the fitness of the firefly
        and changes the best firefly position if it improves it
        
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
        p = Firefly()
        pop.append(p)
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
        pop[j].changePos(pop, j)
        if gBestFitness == -1:
            gBest = pop[j].actualPos
            gBestFitness = pop[j].actualFitness
        else:
            if pop[j].actualFitness < gBestFitness or gBestFitness == -1:
                gBest = pop[j].actualPos
                gBestFitness = pop[j].actualFitness
        print(pop[j].actualPos, pop[j].actualFitness)
    return pop


def runFireflyExperiment(numGen, popSize, fig):
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
        plotFireflies(fig, ax, pop, j)
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
    results.append(runFireflyExperiment(numGen, popSize, fig))
    return results


def plotFireflies(fig, ax, pop, gen):
    """Function that plots the swarm of fireflies
    
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
        ax.plot(gBest[0], gBest[1], markerfacecolor='blue',
                markeredgecolor='red', marker='o', markersize=5)
        ax.set_xlim(minimal, maximum)
        ax.set_ylim(minimal, maximum)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.2)


# Function that plots the memory created through the generations

def plotMem():
    """Function that plots the memory created through the generations
    """    
    global mem
    fig, ax = plt.subplots()
    ax.plot(mem)

    ax.set(xlabel='Gens', ylabel='Fitness')
    ax.grid()

# Plot the optimization function


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


@click.command()
@click.option('--function', "-f", default=None, required=True,
              help=u'Name of the test function to optimize')
@click.option('--dim', "-d", default=2, required=False,
              help=u'Number of dimensions.')
@click.option('--popsize', "-p", default=10, required=False,
              help=u'Size of the population.')
@click.option('--betazero', "-b", default=1, required=False,
              help=u'Initial value of beta.')
@click.option('--gammaval', "-g", default=1, required=False,
              help=u'Initial value of beta.')
@click.option('--ngens', "-n", default=100, required=False,
              help=u'Number of generations.')
def main(function, dim, popsize, betazero, gammaval, ngens):
    global beta_zero, gamma, dimension, minimal, maximum, functionName, step
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
    functionArgv = function
    dimension = int(dim)
    nGens = int(ngens)
    popSize = int(popsize)
    beta_zero = float(betazero)
    gamma = float(gammaval)
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
