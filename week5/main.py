import math
import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import sys

f = 0.99
c = 0.95
mem = []

def generateRandomSolution(dimension, minimal, maximum):
    point = []
    for i in range(0, dimension):
        point.append(random.uniform(minimal, maximum))
    return point


def computeFitness(functionName, point):
    fitness = functionName(point)
    return fitness


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


def initPop(popSize, minimal, maximum):
    pop = []
    for i in range(0, popSize):
        pop.append(generateRandomSolution(2, minimal, maximum))
    return pop

#Function that returns a new vector with the corrected limits
def checkLimits(vec, minimal, maximum):

    vec_new = []
    for i in range(len(vec)):
        if vec[i] < minimal:
            vec_new.append(minimal)
        elif vec[i] > maximum:
            vec_new.append(maximum)
        else:
            vec_new.append(vec[i])
    return vec_new


def recombinateIndividual(xi, v):
    global c
    recombinated = []
    for i in range(len(xi)):
        prob = random.random()
        if c <= prob:
            recombinated.append(v[i])
        else:
            recombinated.append(xi[i])
    return recombinated

#Function that mutates the population and gets the next population of individuals
def mutatePop(pop, functionName, minimal, maximum):
    global f
    vPop = []
    popFitness = []
    for i in range(0, len(pop)):

        popIndex = list(range(0, len(pop)))
        popIndex.remove(i)
        
        #Get three individuals for the operations
        randomIndex = random.sample(popIndex, 3)

        x1 = pop[randomIndex[0]]
        x2 = pop[randomIndex[1]]
        x3 = pop[randomIndex[2]]
        xi = pop[i]

        v = []
        #Do the equation for the mutation v = x1 + F ( x2 - x3)
        for h, j, k in zip(x1, x2, x3):
            v.append(h + f * (j - k))

        #Check the limits to not get exceeded
        v = checkLimits(v, minimal, maximum)

        #Do the "Crossover" with the mutated vector
        vRecombined = recombinateIndividual(xi, v)

        #Get the fitness
        vFitness = computeFitness(functionName, vRecombined)
        xFitness = computeFitness(functionName, xi)

        #Save the best on the next population 
        #If it's equal or less (We are minimizing) then we save the recombined solution
        #If not save the preview solution
        if vFitness <= xFitness:
            vPop.append(vRecombined)
            popFitness.append(vFitness)
        else:
            vPop.append(xi)
            popFitness.append(xFitness)

    return vPop, popFitness


def runDEexperiment(numGen, popSize, functionName, minimal, maximum):
    functionResults = []
    bestFitness = 0
    bestPoint = [0, 0]
    global mem
    #Init the population
    pop = initPop(popSize, minimal, maximum)
    #Iterate thorugh each generation generating a new population based on the mutations
    #of the previous generation
    for j in range(numGen):
        nextPop, nextPopFitness = mutatePop(
            pop, functionName, minimal, maximum)
        bestPopulationIndex = nextPopFitness.index(min(nextPopFitness))
        bestPopulationPoint = nextPop[bestPopulationIndex]
        bestPopulationFitness = min(nextPopFitness)
        #If it's the first iteration save the bestFitness and the bestPoint
        if j == 0:
            bestFitness = bestPopulationFitness
            bestPoint = bestPopulationPoint
        else:
            #If it isnt the first iteration compare to know which one is the best (Get the best solution for all the generations) 
            if bestPopulationFitness < bestFitness:
                bestFitness = bestPopulationFitness
                bestPoint = bestPopulationPoint
        pop = nextPop
        mem.append(bestPopulationFitness)
    functionResults.append([bestFitness, bestPoint[0], bestPoint[1]])
    return functionResults


def runExperiments(numGen, popSize, functionName, functionMin, functionMax, index):
    results = []
    results.append(runDEexperiment(numGen, popSize,
                                   functionName, functionMin[index], functionMax[index]))
    return results


def plotMem():
    global mem
    fig, ax = plt.subplots()
    ax.plot(mem)

    ax.set(xlabel='Gens', ylabel='Fitness')
    ax.grid()


def plotFunction(functionName, minimal, maximum, step):
    fig = plt.figure()
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
    global f, c
    if len(sys.argv) != 6:
        print("Incorrect number of parameters: \n" +
              "\t main.py \"Function Name\" \"nGens\" \"popSize\" \"F\" \"CR\"")
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
    nGens = int(sys.argv[2])
    popSize = int(sys.argv[3])
    f = float(sys.argv[4])
    c = float(sys.argv[5])
    # Parse the arg to function
    functionName = dispatcher[functionArgv]
    # Get the index of the function, to get the minimal, maximum and step for plotting
    index = list(dispatcher.keys()).index(functionArgv)

    abcises = []
    # Append the plot for the function
    abcises.append(plotFunction(
        functionName, functionMin[index], functionMax[index], functionStep[index]))

    obtainedSolutions = runExperiments(
        nGens, popSize, functionName, functionMin, functionMax, index)

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
