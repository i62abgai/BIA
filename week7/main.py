import math
import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import sys
import time
import seaborn

# Optimization functions parameters
functionName = ""
dimension = 0
minimal = 0
maximum = 0
step = 0

# SOMA parameters
prt = 0
pathLength = 0
somaStep = 0
leaderPos = []
leaderFitness = -1

# Memory of the algorithm with
# the best solutions for each migration
mem = []

# Population
pop = []
gen = 0

# Plot values
ax = 0
fig = 0
Z = []
colorBar = True


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
    """[Function that generates a random solution and returns it]

    Returns:
        [np.array] -- [Array of N dimiensions between the maximun and minimal]
    """
    global dimension, minimal, maximum
    point = []
    for i in range(0, dimension):
        point.append(random.uniform(minimal, maximum))
    value = np.asarray(point)
    return value

# Class Particle, each Particle is an object of this class


def generatePTRVector():
    """[Function that generates a pertubation vector]

    Returns:
        [np.array] -- [Binary array of N dimensions called perturbation vector]
    """
    global prt, dimension
    PRTVector = []
    for i in range(dimension):
        r = random.uniform(0, 1)
        PRTVector.append(1) if r < prt else PRTVector.append(0)
    PRTVector = np.asarray(PRTVector)
    return PRTVector


class Particle:
    """[Class particle, each particle saves the actual postion and the actual fitness]
    """

    def __init__(self, i):
        """[
            Constructor of the class
                - Generates an initial random solution for the object particle
                - Saves the fitness of this solution
            ]
        """
        self.ind = i
        self.actualPos = generateRandomSolution()
        self.actualFitness = computeFitness(self.actualPos)

    def move(self):
        """[Function that moves the particle to the end of the path length step by step]
        """
        global somaStep, pathLength, leaderPos
        t = somaStep
        while t < pathLength:
            PRTVector = generatePTRVector()
            if not np.array_equal(PRTVector, np.array([0, 0])):

                # Generate the PTRVector in each jump
                newPos = []
                # Get the new position
                for i in range(len(self.actualPos)):
                    newPos.append(
                        self.actualPos[i] + ((leaderPos[i] - self.actualPos[i]) * t * PRTVector[i]))
                # Check the limits of the new position
                newPos = checkLimits(newPos)
                # Compute the fitness of the new position
                newFitness = computeFitness(newPos)
                # Print the new position in the plot
                time.sleep(0.5)
                plotSwarm(self.ind, newPos)
                # If this new position is better than the actual one save it
                if newFitness <= self.actualFitness:
                    self.actualPos = newPos
                    self.actualFitness = newFitness
                # Sum the step to get the new position
            t += somaStep


def checkLimits(newPos):
    global minimal, maximum
    for i in range(len(newPos)):
        if newPos[i] > maximum:
            newPos[i] = maximum
        elif newPos[i] < minimal:
            newPos[i] = minimal
    return newPos


def computeFitness(point):
    """[Function that computes the fitness]

    Arguments:
        point {[N dimensional point]} -- [Point of N dimensions]

    Returns:
        [float] -- [The fitness for that point (Z value in the function)]
    """
    global functionName
    fitness = functionName(point)
    return fitness

########################################
################ SOMA ##################
########################################


def initPop(popSize):
    """[Function that initializes a population of particles]

    Arguments:
        popSize {[int]} -- [Size of the population]

    Returns:
        [list] -- [List that contains particle objects]
    """
    global minimal, maximum, leaderPos
    pop = []
    for i in range(0, popSize):
        p = Particle(i)
        if i == 0:
            leaderPos = p.actualPos
        pop.append(p)
    return pop


def movePop(pop):
    """[Function that moves the population in each migration]

    Arguments:
        pop {[list]} -- [List that containts the actual population]

    Returns:
        [list] -- [List that contains the next population after the migration]
    """
    global functionName, minimal, maximum, leaderPos, leaderFitness
    lP = leaderPos
    lF = leaderFitness
    for j in range(len(pop)):
        if not np.array_equal(pop[j].actualPos, lP):
            pop[j].move()
            if lF == -1:
                lP = pop[j].actualPos
                lF = pop[j].actualFitness
            else:
                if pop[j].actualFitness < lF or lF == -1:
                    lP = pop[j].actualPos
                    lF = pop[j].actualFitness
        print(pop[j].actualPos, pop[j].actualFitness)

    leaderPos = lP
    leaderFitness = lF
    return pop


def runSOMAExperiment(numGen, popSize):
    functionResults = []
    global mem, leaderPos, leaderFitness, functionName, minimal, maximum, ax, fig, pop, gen, Z
    X, Y, Z = getMesh()
    ax = fig.add_subplot()
    # Init the population
    pop = initPop(popSize)
    # Iterate thorugh each generation generating a new population based on the mutations
    # of the previous generation
    for j in range(numGen):
        nextPop = movePop(pop)
        gen = j
        pop = nextPop
        mem.append(leaderFitness)
        print("\n\nGEN: ", j, ": ", leaderPos, leaderFitness, "\n\n")
    functionResults.append([leaderFitness, leaderPos[0], leaderPos[1]])
    return functionResults

# For each function run the experiment


def runExperiments(numGen, popSize):
    results = []
    results.append(runSOMAExperiment(numGen, popSize))
    return results


def plotSwarm(i, newPos):
    global minimal, maximum, leaderPos, ax, fig, pop, gen, Z, colorBar
    if plt.fignum_exists(1):
        ax.clear()
        plt.imshow(Z, extent=[minimal, maximum, minimal, maximum], origin='lower', cmap='coolwarm')
        title = "gen " + str(gen)
        fig.suptitle(title)
        for j in range(len(pop)):
            p = "p"+str(j)
            if j == i:
                ax.plot(newPos[0], newPos[1], markerfacecolor='yellow',
                        markeredgecolor='green', marker='o', markersize=5)
                ax.text(newPos[0], newPos[1], p)
            else:
                ax.plot(pop[j].actualPos[0], pop[j].actualPos[1], markerfacecolor='green',
                        markeredgecolor='blue', marker='o', markersize=5)
                ax.text(pop[j].actualPos[0], pop[j].actualPos[1], p)
        ax.plot(leaderPos[0], leaderPos[1], markerfacecolor='blue',
                markeredgecolor='red', marker='D', markersize=5)
        if colorBar == True:
            plt.colorbar()
            colorBar = False
        ax.set_xlim(minimal, maximum)
        ax.set_ylim(minimal, maximum)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.2)

# Function that plots the memory created through the generations


def plotMem():
    global mem
    fig, ax = plt.subplots()
    ax.plot(mem)

    ax.set(xlabel='Gens', ylabel='Fitness')
    ax.grid()


# Plot the optimization function

def getMesh():
    global minimal, maximum, step
    X = np.arange(minimal, maximum, step)
    Y = np.arange(minimal, maximum, step)
    X, Y = np.meshgrid(X, Y)
    z_rows = []
    for xi, y_i in zip(X, Y):
        point = [xi, y_i]
        z_rows.append(functionName(point))
    Z = np.array(z_rows)
    return X, Y, Z


def plotFunction(figure):
    """[Function that prints the heatmap]

    Arguments:
        figure {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    global minimal, maximum, functionName, step, fig, Z
    axPlot = plt.axes(projection="3d")

    X, Y, Z = getMesh()
    surf = axPlot.plot_surface(X, Y, Z, cmap='coolwarm',
                               linewidth=0, antialiased=False, alpha=0.35)
    figure.colorbar(surf, shrink=0.5, aspect=5)

    return axPlot


def main():
    global prt, pathLength, somaStep, dimension, minimal, maximum, functionName, step, leaderPos, leaderFitness, fig
    if len(sys.argv) != 8:
        print("Incorrect number of parameters: \n" +
              "\t main.py \"Function Name\" \"dimension\" \"nGens\" \"popSize\" \"prt\" \"pathLength\" \"step\"")
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

    # Adjust all the arguments to the parameters
    functionArgv = sys.argv[1]
    dimension = int(sys.argv[2])
    nGens = int(sys.argv[3])
    popSize = int(sys.argv[4])
    prt = float(sys.argv[5])
    pathLength = int(sys.argv[6])
    somaStep = float(sys.argv[7])

    # Parse the arg to function
    functionName = dispatcher[functionArgv]

    # Get the index of the function, to get the minimal, maximum and step for plotting
    index = list(dispatcher.keys()).index(functionArgv)
    minimal = functionMin[index]
    maximum = functionMax[index]
    step = functionStep[index]
    plt.ion()
    fig = plt.figure(1)
    obtainedSolutions = runExperiments(nGens, popSize)
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
