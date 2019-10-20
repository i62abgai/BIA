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
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def euclideanDistance(self, city):
        x = abs(self.x - city.x)**2
        y = abs(self.y - city.y)**2
        return np.sqrt(x+y)


def getSolDistance(sol):
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
    fitness = 1/float(getSolDistance(sol))
    return fitness


def generateRandomSolution():
    global n_nodes
    perm = np.random.permutation(n_nodes)
    return perm


def initPop(popSize):
    global n_nodes
    pop = []
    for i in range(0, popSize):
        pop.append(generateRandomSolution())
    return pop


def popFitness(pop):
    results = {}
    for i in range(len(pop)):
        results[i] = computeFitness(pop[i])
    return sorted(results.items(), key=operator.itemgetter(1), reverse=True)


def selection(rankedPop, elite):
    selecResults = []
    for i in range(0, elite):
        selecResults.append(rankedPop[i][0])

    for i in range(0, len(rankedPop)-elite):
        rand = random.randint(0, len(rankedPop)-1)
        selecResults.append(rankedPop[rand][0])

    return selecResults


def fatherSelection(pop, selecResults):
    fathers_pop = []
    for i in range(0, len(selecResults)):
        index = selecResults[i]
        fathers_pop.append(pop[index])
    return fathers_pop


def crossover(p1, p2):
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
    offspring = []
    for i in range(0, elite):
        offspring.append(fathers_pop[i])
    for i in range(0, len(fathers_pop) - elite):
        child = crossover(fathers_pop[i], fathers_pop[len(fathers_pop)-i-1])
        offspring.append(child)
    return offspring


def mutate(sol):
    pointA = random.randint(0, len(sol)-1)
    pointB = random.randint(0, len(sol)-1)
    aux = sol[pointA]
    sol[pointA] = sol[pointB]
    sol[pointB] = aux
    return sol


def mutatePop(pop, mutationRate):
    mutatedPop = []
    for i in range(0, len(pop)):
        if random.random() < mutationRate:
            mutatedSol = mutate(pop[i])
            mutatedPop.append(mutatedSol)
        else:
            mutatedPop.append(pop[i])
    return mutatedPop


def nextPop(currentGeneration, elite, mutationRate):
    rankedPop = popFitness(currentGeneration)

    selectResults = selection(rankedPop, elite)
    fathers_pop = fatherSelection(currentGeneration, selectResults)
    offspring = createOffspring(fathers_pop, elite)
    nextGen = mutatePop(offspring, mutationRate)
    return nextGen


def readInstance(fileName):
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
        axs[1].annotate(txt, (x[i],y[i]))
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
    
    print(" Population Size : ", popSize, "\n Elite Size : ", eliteSize, "\n Mutation Rate : ", mutationRate, "\n Number of Generations : ", numGen)
    plt.show()

if __name__ == "__main__":
    main()
