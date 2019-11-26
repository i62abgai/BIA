import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import pandas as pd
import sys

adjacencyMatrix = []
phMatrix = []
n_nodes = 0
city_list = []
alpha = beta = 0
v = 0


class Ant:
    def __init__(self, start, i):
        self.antIndex = i
        self.antFitness = 0
        self.actualPos = start
        self.visitedCities = []
        self.visitedCities.append(start)

    def move(self):
        global phMatrix, adjacencyMatrix, city_list, alpha, beta
        probs = []
        ind = []
        denominator = 0
        for i, c in enumerate(city_list):
            if i not in self.visitedCities:
                denominator += (np.power(phMatrix[self.actualPos][i], alpha)) * (
                    1/(np.power(adjacencyMatrix[self.actualPos][i], beta)))
        for i, c in enumerate(city_list):
            if i not in self.visitedCities:
                numerator = (np.power(phMatrix[self.actualPos][i], alpha)) * (
                    1/np.power(adjacencyMatrix[self.actualPos][i], beta))
                probs.append(numerator/denominator)
                ind.append(i)
        np_cumulative = np.cumsum(probs)
        r = random.uniform(0, 1)

        city_selected = 0
        for i, c in enumerate(np_cumulative):
            if r < c:
                city_selected = ind[i]
                break
        self.visitedCities.append(city_selected)
        self.antFitness = getSolDistance(self.visitedCities)


def vaporize(movements, pop):
    global phMatrix, v
    for i, m in enumerate(movements):
        phMatrix[m[0]][m[1]] = phMatrix[m[0]][m[1]] + (1/pop[i].antFitness)
        phMatrix[m[1]][m[0]] = phMatrix[m[1]][m[0]] + (1/pop[i].antFitness)
    for i, r in enumerate(phMatrix):
        for j, c in enumerate(phMatrix[i]):
            phMatrix[i][j] = phMatrix[i][j] * v


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


def generateRandomSolution(n_ants):
    perm = np.random.permutation(n_ants)
    return perm


def initPop(n_ants):
    global n_nodes
    pop = []
    randInit = generateRandomSolution(n_ants)
    for i in range(0, len(randInit)):
        a = Ant(randInit[i], randInit[i])
        pop.append(a)
    return pop


def readInstance(fileName):
    global n_nodes, adjacencyMatrix, phMatrix, city_list
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
    phMatrix = np.ones((n_nodes, n_nodes))


def getBest(pop):
    best = pop[0]
    for i, a in enumerate(pop):
        if pop[i].antFitness < best.antFitness:
            best = pop[i]
    return best


def runAntsexperiment(n_ants, gens):
    global n_nodes
    bestSol = -1
    bestSolFitness = -1
    mem = []
    for n in range(gens):
        print("\tGEN: ", n)
        pop = initPop(n_ants)
        for i in range(n_nodes-1):
            # All ants move to a new node
            print("\t\tANT MOVEMENT: ", i)
            movements = []
            for j in range(len(pop)):
                # Ant move
                pop[j].move()
                print(j, " --> ", pop[j].visitedCities,
                      getSolDistance(pop[j].visitedCities))
                movements.append((pop[j].visitedCities[len(
                    pop[j].visitedCities)-2], pop[j].visitedCities[len(pop[j].visitedCities)-1]))
        # Vaporization
        vaporize(movements, pop)
        #Calculate best solution for each gen
        bestPopSol = getBest(pop)
        mem.append(bestPopSol.antFitness)
        if bestPopSol.antFitness <= bestSolFitness or bestSolFitness == -1:
            bestSol = bestPopSol.visitedCities
            bestSolFitness = bestPopSol.antFitness

    return bestSol, mem


def plotResults(mem, bestSol):
    global city_list
    fig, axs = plt.subplots(2)
    plt.title(str(bestSol))
    fig.suptitle('Results obtained from Ant colony')
    axs.flat[0].set(xlabel='Nodes', ylabel='Distance')
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
    global alpha, beta, v
    if len(sys.argv) < 6:
        print("Incorrect number of parameters: \n" +
              "\t main.py \"instance file\" \"n ants\" ")
        return -1

    readInstance(sys.argv[1])
    n_ants = int(sys.argv[2])
    gens = int(sys.argv[3])
    alpha = float(sys.argv[4])
    beta = float(sys.argv[5])
    v = float(sys.argv[6])

    bestSol, mem = runAntsexperiment(n_ants, gens)
    plotResults(mem, bestSol)
    # print(" Population Size : ", popSize, "\n Elite Size : ", eliteSize, "\n Mutation Rate : ", mutationRate, "\n Number of Generations : ", numGen)
    plt.show()


if __name__ == "__main__":
    main()
