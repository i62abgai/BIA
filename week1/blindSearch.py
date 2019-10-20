import math
import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

class solGenerator:
    def generateRandomSolution(self, dimension, min, max):
        point = []
        for i in range(0,dimension):
            point.append(random.randrange(min,max,1))
        return point

class evaluator:
    def computeFitness(self, functionName, point):
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
        sum+=np.sin(value)*np.power(np.sin((i*(value**2))/math.pi),2*10)
        i+=1
    return -sum


def testRosenbrockFunction(points):
    l=len(points)
    for index, obj in enumerate(points):
        if index == 0:
            sum = 0  
        if index < (l - 1):
            sum+=100*(points[index+1] - (obj)**2)**2 + (obj-1)**2    
    return sum


def testZakharovFunction(points):
    firstSum = 0
    secondSum = 0
    i = 1
    for value in points:
        firstSum += value**2
        secondSum += 0.5*i*value
        i+=1
    return firstSum + np.power(secondSum,2) + np.power(secondSum,4)


def getW(x):
    return 1+(x-1)/4


def testLevyFunction(points):
    sum = 0
    part1 = np.sin(math.pi*getW(points[0]))**2
    wd = getW(points[len(points)-1])
    for index, value in enumerate(points):
        wi = getW(value)
        if index < (len(points)-1):
            sum+=( (wi-1)**2 ) * ( 1+10*( np.sin(math.pi*wi+1)**2 ) ) + ( (wd-1)**2 ) * ( 1+np.sin( 2*math.pi*wd )**2 )
    return part1 + sum


def plotFunction(functionName, min, max, step):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    X = np.arange(min, max, step)
    Y = np.arange(min, max, step)
    X, Y = np.meshgrid(X, Y)
    Z=[]
    z_rows=[]
    for x_i, y_i in zip(X,Y):
        point = [x_i, y_i]
        z_rows.append(functionName(point))
    Z=np.array(z_rows)
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    return ax


def runARandomSearchExperiment(numPopulation, numSolutions, functionName, min, max):
    functionResults = []
    eval = evaluator()
    sGen = solGenerator()
    bestFitness = 0
    bestPoint = 0
    for j in range(numPopulation):
        bestPopulationPoint = sGen.generateRandomSolution(2, min, max)
        bestPopulationFitness = eval.computeFitness(functionName, bestPopulationPoint)
        for i in range(numSolutions):
            newPoint = sGen.generateRandomSolution(2, min, max)
            newFitness = eval.computeFitness(functionName, newPoint)
            if newFitness < bestPopulationFitness:
                bestPopulationPoint = newPoint
                bestPopulationFitness = newFitness
        if j == 0:
            bestFitness = bestPopulationFitness
            bestPoint = bestPopulationPoint
        else:
            if bestPopulationFitness < bestFitness:
                bestFitness = bestPopulationFitness
                bestPoint = bestPopulationPoint
    functionResults.append([bestFitness, bestPoint[0], bestPoint[1]])
    return functionResults


def runExperiments(numPopulation, numSolutions, functionNames, functionMin, functionMax):
    results = []
    for index, name in enumerate(functionNames):
        results.append(runARandomSearchExperiment(numPopulation, numSolutions, name, functionMin[index], functionMax[index]))
    return results


def main():
    functionNames = [testSphereFunction, testAckleyFunction, testGriewankFunction,
    testRastriginFunction, testSchwefelFunction, testMichalewiczFunction,
    testRosenbrockFunction, testZakharovFunction, testLevyFunction]
    functionMin = [-6, -33, -600, -6, -500, 0, -10, -10, -10]
    functionMax = [6, 33, 600, 6, 500, 4, 10, 10, 10]
    functionStep = [0.01, 0.25, 1, 0.01, 1, 0.01, 0.25, 0.25, 0.25]
    
    abcises = []
    for index, name in enumerate(functionNames):
        abcises.append(plotFunction(name,functionMin[index], functionMax[index], functionStep[index]))
    
    obtainedSolutions = runExperiments( 10, 10, functionNames, functionMin, functionMax)

    for functionSols, ax in zip(obtainedSolutions, abcises):
        x_sol = []
        y_sol = []
        z_sol = []
        for point in functionSols:
            z_sol.append(point[0])
            x_sol.append(point[1])
            y_sol.append(point[2])
        
        ax.plot(x_sol, y_sol, z_sol, markerfacecolor='green', markeredgecolor='blue', marker='o', markersize=10)
    plt.show()

if __name__ == "__main__":
    main()