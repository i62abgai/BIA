import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import pandas as pd
import sys

adjacencyMatrix = []
n_nodes = 0
city_list = []


class City:
    """Class list with two attributes:
        - x : value in the x axis
        - y : value in the y axis
    """

    def __init__(self, x, y):
        """Constructor of the class. Assigns the values x and y for the city

        Arguments:
            x {[float]} -- [value in the x axis]
            y {[float]} -- [value in the y axis]
        """
        self.x = x
        self.y = y

    def euclideanDistance(self, city):
        """Function that calculates the euclidean between the object city, and another city

        Arguments:
            city {[City]} -- [City object]

        Returns:
            [float] -- [Euclidean distance]
        """
        x = abs(self.x - city.x)**2
        y = abs(self.y - city.y)**2
        return np.sqrt(x+y)


def getSolDistance(sol):
    """Function that calculates the total distance between all the cities in a solution

    Arguments:
        sol {[list]} -- [List of City objects]

    Returns:
        [float] -- [Distance between all the cities in a solution]
    """
    global adjacencyMatrix
    distance = 0
    for i in range(len(sol)-1):
        fromCity = sol[i]
        toCity = sol[i + 1]
        distance += adjacencyMatrix[fromCity][toCity]
    fromCity = sol[len(sol)-1]
    toCity = sol[0]
    distance += adjacencyMatrix[fromCity][toCity]
    return distance


def computeFitness(sol):
    """Function that calculates the fitness for a solution.
    Because I wanted to maximize instead of minimizing I divide 1/distance

    Arguments:
        sol {[list]} -- [List of City objects]

    Returns:
        [float] -- [1 divided by the total distance of the solution]
    """
    fitness = 1/float(getSolDistance(sol))
    return fitness


def generateRandomSolution():
    """Function that generates a random solution (Permutation)

    Returns:
        [list] -- [List with a random permutation]
    """
    global n_nodes
    perm = np.random.permutation(n_nodes)
    return perm


def initPop(popSize):
    """Function that initializes a population with random solutions

    Arguments:
        popSize {[int]} -- [Size of the population]

    Returns:
        [list] -- [List of random solutions]
    """
    global n_nodes
    pop = []
    for i in range(0, popSize):
        pop.append(generateRandomSolution())
    return pop


def popFitness(pop):
    """Function that calculates the fitness for all the solutions in the population

    Arguments:
        pop {[list]} -- [List of solutions]

    Returns:
        [list] -- [Returns the population with the new fitness, sorted from the best to the worst]
    """
    results = {}
    for i in range(len(pop)):
        results[i] = computeFitness(pop[i])
    return sorted(results.items(), key=operator.itemgetter(1), reverse=True)


def selection(rankedPop, elite):
    """Function that selects the new population. 
    First we add the elite members, then we select randomly other solutions in the population.

    Arguments:
        rankedPop {[list]} -- [Population sorted from best to worst]
        elite {[int]} -- [Size of the elite]

    Returns:
        [list] -- [Selected population]
    """
    selecResults = []
    for i in range(0, elite):
        selecResults.append(rankedPop[i][0])

    for i in range(0, len(rankedPop)-elite):
        rand = random.randint(0, len(rankedPop)-1)
        selecResults.append(rankedPop[rand][0])

    return selecResults


def fatherSelection(pop, selecResults):
    """Function that selects the fathers to do the crossover operator

    Arguments:
        pop {[list]} -- [Population of solutions]
        selecResults {[list]} -- [List with the selected solutions of the population]

    Returns:
        [list] -- [List with the fathers selected]
    """
    fathers_pop = []
    for i in range(0, len(selecResults)):
        index = selecResults[i]
        fathers_pop.append(pop[index])
    return fathers_pop


def crossover(p1, p2):
    """Crossover operator, which crosses two fathers and gets a new child

    Arguments:
        p1 {[sol]} -- [Father number 1]
        p2 {[sol]} -- [Father number 2]

    Returns:
        [sol] -- [Crossover solution between father 1 and father 2,
                    if the new solution is worst then return the father 1]
    """
    child = []
    childP1 = []
    childP2 = []

    randPoint_1 = int(random.random() * len(p1))
    randPoint_2 = int(random.random() * len(p2))

    startPoint = min(randPoint_1, randPoint_2)
    endPoint = max(randPoint_1, randPoint_2)

    for i in range(startPoint, endPoint):
        childP1.append(p1[i])

    childP2 = [item for item in p2 if item not in childP1]

    child = childP1 + childP2
    if computeFitness(child) > computeFitness(p1):
        return child
    else:
        return p1


def createOffspring(fathers_pop, elite):
    """Function that creates the new offspring for the population selected
    
    Arguments:
        fathers_pop {[list]} -- [List of father selected]
        elite {[int]} -- [Size of the elite]
    
    Returns:
        [list] -- [List with the solutions of the offspring]
    """
    offspring = []
    for i in range(0, elite):
        offspring.append(fathers_pop[i])
    for i in range(0, len(fathers_pop) - elite):
        child = crossover(fathers_pop[i], fathers_pop[len(fathers_pop)-i-1])
        offspring.append(child)
    return offspring


def mutate(sol):
    """Function that mutates a solution
    
    Arguments:
        sol {[list]} -- [List of City objects]
    
    Returns:
        [sol] -- [New solution mutated]
    """    
    pointA = random.randint(0, len(sol)-1)
    pointB = random.randint(0, len(sol)-1)
    aux = sol[pointA]
    sol[pointA] = sol[pointB]
    sol[pointB] = aux
    return sol


def mutatePop(pop, mutationRate):
    """Function that mutates the solutions of the population with some probability
    
    Arguments:
        pop {[list]} -- [List of solutions]
        mutationRate {[float]} -- [Probability of mutation]
    
    Returns:
        [list] -- [New population mutated]
    """    
    mutatedPop = []
    for i in range(0, len(pop)):
        if random.random() < mutationRate:
            mutatedSol = mutate(pop[i])
            mutatedPop.append(mutatedSol)
        else:
            mutatedPop.append(pop[i])
    return mutatedPop


def nextPop(currentGeneration, elite, mutationRate):
    """Function that creates the next population:
        - Select solutions of the actual population
        - Select the fathers for the next population
        - Create the offspring
        - Mutate the offspring
    
    Arguments:
        currentGeneration {[list]} -- [List of actual solutions]
        elite {[int]} -- [Size of the elite]
        mutationRate {[float]} -- [Probability of mutation]
    
    Returns:
        [list] -- [Population for the next generation]
    """    
    rankedPop = popFitness(currentGeneration)

    selectResults = selection(rankedPop, elite)
    fathers_pop = fatherSelection(currentGeneration, selectResults)
    offspring = createOffspring(fathers_pop, elite)
    nextGen = mutatePop(offspring, mutationRate)
    return nextGen


def readInstance(fileName):
    """Function that reads a file with the cities and its coordinates, and calculates the adjacency matrix
    
    Arguments:
        fileName {[string]} -- [Name of the file where the instance is saved]
    """    
    global n_nodes
    global adjacencyMatrix
    global city_list
    data = pd.read_csv(fileName, header=None)
    lol = data.values.tolist()
    listOfCities = []
    for val in lol:
        val_city = City(val[0], val[1])
        listOfCities.append(val_city)
    n_nodes = len(lol)
    adjacencyMatrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                adjacencyMatrix[i][j] = listOfCities[i].euclideanDistance(
                    listOfCities[j])
    city_list = listOfCities


def runGAexperiment(popSize, eliteSize, mutationRate, n_gens):
    """Function that runs the genetical algorithm experiment
    
    Arguments:
        popSize {[int]} -- [Size of the popultion]
        eliteSize {[int]} -- [Size of the elite]
        mutationRate {[float]} -- [Probability of mutation]
        n_gens {[int]} -- [Number of generations]
    
    Returns:
        [sol] -- [Returns the best solution obtained in all generations]
    """    
    global n_nodes
    mem = []
    pop = initPop(popSize)
    mem.append(1/popFitness(pop)[0][1])
    for i in range(0, n_gens):
        pop = nextPop(pop, eliteSize, mutationRate)
        mem.append(1/popFitness(pop)[0][1])
    bestSol = pop[popFitness(pop)[0][0]]
    plotResults(mem, bestSol)
    return bestSol


def plotResults(mem, bestSol):
    """Function that plots the results of the algorithm and draws the function in all values
    
    Arguments:
        mem {[list]} -- [List with the best fitness for each generation]
        bestSol {[sol]} -- [Best solution obtained in all generations]
    """ 
    global city_list
    fig, axs = plt.subplots(2)
    plt.title(str(bestSol))
    fig.suptitle('Results obtained from GA')
    axs.flat[0].set(xlabel='Generations', ylabel='Distance')
    axs[0].plot(mem)
    x = []
    y = []
    n = []
    for i in range(0, len(bestSol)):
        c = city_list[bestSol[i]]
        x.append(c.x)
        y.append(c.y)
        n.append(bestSol[i])
    c = city_list[bestSol[0]]
    x.append(c.x)
    y.append(c.y)
    n.append(bestSol[0])
    for i, txt in enumerate(n):
        axs[1].annotate(txt, (x[i], y[i]))
    axs[1].plot(x, y)
    axs[1].plot(x, y, 'ro')
    axs.flat[1].set(xlabel='X', ylabel='Y')
    for ax in axs.flat:
        ax.grid(True)


def main():
    if len(sys.argv) != 2:
        print("Incorrect number of parameters: \n" +
              "\t main.py \"instance file\"")
        return -1

    readInstance(sys.argv[1])
    popSize = 100
    eliteSize = 3
    mutationRate = 0.01
    numGen = 20
    bestSol = runGAexperiment(popSize, eliteSize, mutationRate, numGen)

    print(" Population Size : ", popSize, "\n Elite Size : ", eliteSize,
          "\n Mutation Rate : ", mutationRate, "\n Number of Generations : ", numGen)
    plt.show()


if __name__ == "__main__":
    main()
