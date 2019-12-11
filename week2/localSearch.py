import math
import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

class solGenerator:
    """[Class with only one function that allows to generate random solutions]
    """
    def generateRandomSolution(self, dimension, min, max):
        """Function that generates a random solution between bounds and with a determined dimension

        Arguments:
            dimension {[int]} -- [Dimension of the problem (In 2D problems, there are 2 dimensions)]
            minimum {[float]} -- [Minimum bound]
            maximum {[float]} -- [Maximum bound]

        Returns:
            [list] -- [Solution with described dimension between the bounds]
        """
        point = []
        for i in range(0,dimension):
            point.append(random.randint(min,max))
        return point

class evaluator:
    """[Class with one function that evaluates the fitness of a solution] 
    """    
    def computeFitness(self, functionName, point):
        """Function that evaluates the fitness of a solution and returns it
        
        Arguments:
            functionName {[function]} -- [Name of the function that evaluates the fitness]
            point {[list]} -- [Solution with n dimensions]
        
        Returns:
            [float] -- [Value of the solution evaluated with the function]
        """  
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
    """Function that plots the results of the algorithm and draws the function in all values
    
    Arguments:
        functionName {[function]} -- [Name of the function that evaluates the fitness]
        minimum {[float]} -- [Minimum bound]
        maximum {[float]} -- [Maximum bound]
        step {[type]} -- [Step to have a good resolution in the plotting, the smaller the better]
    
    Returns:
        [axis] -- [Returns the axis plot of the function]
    """ 
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
    plt.title(functionName)
    plt.ylabel('y')
    plt.xlabel('x')
    return ax

def runSAExperiment(numPopu, maxIterations, functionName, min, max):
    """Function that runs the simulated annealing experiment
    
    Arguments:
        numPopu {[int]} -- [Size of the popultion]
        maxIterations {[int]} -- [Number of maximum iterations]
        functionName {[function]} -- [Function name which evaluates the fitness]
        minimum {[float]} -- [Minimum bound]
        maximum {[float]} -- [Maximum bound]
    
    Returns:
        [list] -- [List of solutions at the end of the algorithm]
    """     
    functionResults = []
    eval = evaluator()
    sGen = solGenerator()
    bestFitness = bestPoint = 0
    sigma = 2
    temperature = 200
    alpha = 0.98
    
    #For each iteration we generate only one solution
    for j in range(numPopu):
        bestPopulationPoint = sGen.generateRandomSolution(2, min, max)
        bestPopulationFitness = eval.computeFitness(functionName, bestPopulationPoint)
        #Explore a delimited number of neighbours
        for i in range(maxIterations):
            newNeighbor = [np.random.normal(bestPopulationPoint[0], sigma),np.random.normal(bestPopulationPoint[1], sigma)]
            newFitness = eval.computeFitness(functionName, newNeighbor)
            #We change the best fitness if the fitness if the neighbor is better
            f = newFitness - bestPopulationFitness
            if f < 0:
                bestPopulationPoint = newNeighbor
                bestPopulationFitness = newFitness
            #If the new fitness is not better with some probability (depends on the temperature)
            # we could accept this new solution, even if its worse
            else:
                r = random.randint(1,1000)/1000
                e = math.exp(-f/temperature)
                if r < e:
                    bestPopulationPoint = newNeighbor
                    bestPopulationFitness = newFitness
                    temperature = temperature * alpha                
        #For each population we save the best one
        if j == 0:
            bestFitness = bestPopulationFitness
            bestPoint = bestPopulationPoint
        else:
            if bestPopulationFitness < bestFitness:
                bestFitness = bestPopulationFitness
                bestPoint = bestPopulationPoint
    #We return all the best solutions for each function
    functionResults.append([bestFitness, bestPoint[0], bestPoint[1]])
    return functionResults

def runALSExperiment(numPopu, numNeighbours, functionName, min, max):
    """Function that runs the local search experiment
    
    Arguments:
        numPopu {[int]} -- [Size of the popultion]
        numNeighbours {[int]} -- [Size of the neighbor]
        functionName {[function]} -- [Function name which evaluates the fitness]
        minimum {[float]} -- [Minimum bound]
        maximum {[float]} -- [Maximum bound]
    
    Returns:
        [list] -- [List of solutions at the end of the algorithm]
    """ 
    functionResults = []
    eval = evaluator()
    sGen = solGenerator()
    bestFitness = 0
    bestPoint = 0
    sigma = 2
    #For each iteration we generate only one solution
    for j in range(numPopu):
        bestPopulationPoint = sGen.generateRandomSolution(2, min, max)
        bestPopulationFitness = eval.computeFitness(functionName, bestPopulationPoint)
        #Explore a delimited number of neighbours
        for i in range(numNeighbours):
            newNeighbor = [np.random.normal(bestPopulationPoint[0], sigma),np.random.normal(bestPopulationPoint[1], sigma)]
            newFitness = eval.computeFitness(functionName, newNeighbor)
            #We change the best fitness if the fitness if the neighbor is better
            if newFitness < bestPopulationFitness:
                bestPopulationPoint = newNeighbor
                bestPopulationFitness = newFitness
        #For each iteration we save the best one
        if j == 0:
            bestFitness = bestPopulationFitness
            bestPoint = bestPopulationPoint
        else:
            if bestPopulationFitness < bestFitness:
                bestFitness = bestPopulationFitness
                bestPoint = bestPopulationPoint
    #We return all the best solutions for each function
    functionResults.append([bestFitness, bestPoint[0], bestPoint[1]])
    return functionResults


def runExperiments(numPopu, numNeighbours, functionNames, functionMin, functionMax):
    """Function that runs all the experiments for each function
    
    Arguments:
        numPopulation {[int]} -- [Size of the popultion]
        numSolutions {[int]} -- [Size of the neighbor]
        functionNames {[function]} -- [Function name which evaluates the fitness]
        functionMin {[float]} -- [Minimum bound]
        functionMax {[float]} -- [Maximum bound]
    
    Returns:
        [list] -- [List of Solutions obtained for each function]
    """  
    results = []
    als = []
    sa = []
    for index, name in enumerate(functionNames):
        als.append(runALSExperiment(numPopu, numNeighbours, name, functionMin[index], functionMax[index]))
        sa.append(runSAExperiment(numPopu, numNeighbours, name, functionMin[index], functionMax[index]))
    results.append(als)
    results.append(sa)
    return results


def main():
    functionNames = [testSphereFunction, testAckleyFunction, testGriewankFunction,
    testRastriginFunction, testSchwefelFunction,
    testRosenbrockFunction, testZakharovFunction, testLevyFunction]
    functionMin = [-6, -33, -600, -6, -500, -10, -10, -10]
    functionMax = [6, 33, 600, 6, 500, 10, 10, 10]
    functionStep = [0.01, 0.25, 1, 0.01, 1, 0.25, 0.25, 0.25]
    
    abcises = []
    for index, name in enumerate(functionNames):
        abcises.append(plotFunction(name,functionMin[index], functionMax[index], functionStep[index]))
    
    obtainedSolutions = runExperiments( 10, 10, functionNames, functionMin, functionMax)
    for i, algorithm in enumerate(obtainedSolutions):
        for functionSols, ax in zip(algorithm, abcises):
            x_sol = []
            y_sol = []
            z_sol = []
            for point in functionSols:
                z_sol.append(point[0])
                x_sol.append(point[1])
                y_sol.append(point[2])
            if i == 0:
                ax.plot(x_sol, y_sol, z_sol, markerfacecolor='green', markeredgecolor='blue', marker='o', markersize=10)
            else:
                ax.plot(x_sol, y_sol, z_sol, markerfacecolor='yellow', markeredgecolor='red', marker='o', markersize=10)
    plt.show()

if __name__ == "__main__":
    main()