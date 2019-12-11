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
from operator import attrgetter

mutateRate = 0
minimum = 0
maximum = 0


class Sol():
    """Class solution which has threes attributes:
        - x : x value
        - f1_ : fitness of the value x in the function1
        - f2_ : fitness of the value x in the function2
    """    
    def __init__(self, x):
        """Constructor of the class
        
        Arguments:
            x {[float]} -- [x value to initialize the sol object]
        """        
        self.x_ = x
        self.f1_ = function1(x)
        self.f2_ = function2(x)

    def computeSolFitness(self):
        """Function that computes the fitness of the x value for both functions
        """        
        self.f1_ = function1(self.x_)
        self.f2_ = function2(self.x_)


def function1(x):
    """Function 1
    
    Arguments:
        x {[float]} -- [x value]
    
    Returns:
        [float] -- [fitness of the value x in the function1]
    """    
    result = - np.power(x, 2)
    return result


def function2(x):
    """Function 1
    
    Arguments:
        x {[float]} -- [x value]
    
    Returns:
        [float] -- [fitness of the value x in the function2]
    """    
    result = (x - 2)
    result = - np.power(result, 2)
    return result


def initPop(popSize):
    """Function that initializes the population of solutions
    
    Arguments:
        popSize {[int]} -- [Size of the population]
    
    Returns:
        [list] -- [List of solutions (particles)]
    """ 
    global minimum, maximum
    xValues = random.sample(range(minimum, maximum), popSize)
    #xValues = [-2, -1, 0, 2, 4, 1]
    p = []
    for i in xValues:
        s = Sol(i)
        p.append(s)
    return p


def evaluatePop(pop):
    """Function that evaluates the fitness of both functions for each solution
    
    Arguments:
        pop {[list]} -- [List of solutions]
    
    Returns:
        [list] -- [List of evaluated solutions]
    """    
    for i in range(len(pop)):
        pop[i].computeSolFitness()
    return pop


def isDominating(s1, s2):
    """Function that checks if a solution is dominating another solution
    
    Arguments:
        s1 {[sol]} -- [Sol object to check if dominates]
        s2 {[type]} -- [Sol object to check if is dominated]
    
    Returns:
        [bool] -- [Retuns if s1 is dominating s2, False in other case]
    """    
    t = False
    if ((s1.f1_ >= s2.f1_) and (s1.f2_ > s2.f2_)) or ((s1.f1_ > s2.f1_) and (s1.f2_ >= s2.f2_)):
        t = True
    return t


def fastNonDominated(pop, n, s):
    """Function that executes the Fast Non Dominated Sorting for a population
    
    Arguments:
        pop {[list]} -- [List of solutions]
        n {[list]} -- [List that contains how many dominates x value]
        s {[list]} -- [List of list that contains which one dominates]
    
    Returns:
        [list] -- [Returns the pareto in rank 1]
    """    
    Q = []
    for i in range(len(pop)):
        for j in range(len(pop)):
            if i != j:
                if isDominating(pop[i], pop[j]):
                    # print("\t",pop[i].f1_, pop[i].f2_, " dominates ", pop[j].f1_, pop[j].f2_)
                    s[i].append(j)
                if isDominating(pop[j], pop[i]):
                    n[i] += 1

    while not all(j < 0 for j in n):
        q = []
        for i in range(len(n)):
            if n[i] == 0:
                q.append(i)
                for y in s[i]:
                    n[y] -= 1
                n[i] -= 1
        Q.append(q)

    return Q


def sortValues(pop, q, f):
    """Function that sorts a list based in the order of the fitness 1 or fitness 2
    
    Arguments:
        pop {[list]} -- [List of solutions]
        q {[list]} -- [List which is used to order]
        f {[string]} -- [Indicates if it is order based in fitness 1 or fitness 2]
    
    Returns:
        [type] -- [description]
    """    
    sortedl = []
    l = []
    if f == 'f1':
        for i in pop:
            l.append(i.f1_)
    else:
        for i in pop:
            l.append(i.f2_)

    while(len(sortedl) != len(q)):
        if l.index(min(l)) in q:
            sortedl.append(l.index(min(l)))
        l[l.index(min(l))] = math.inf
    return sortedl


def getCrowding(pop, n, s, q):
    """Function that calculates the crowding distance
    
    Arguments:
        pop {[list]} -- [List of solutions]
        n {[list]} -- [List that contains how many dominates x value]
        s {[list]} -- [List of list that contains which one dominates]
        q {[list]} -- [pareto]
    
    Returns:
        [type] -- [description]
    """    
    nonDominated = q
    d = [0 for i in range(len(nonDominated))]
    sortedf1 = sortValues(pop, nonDominated, 'f1')
    sortedf2 = sortValues(pop, nonDominated, 'f2')

    d[0] = d[len(d)-1] = 999999

    for i in range(1, len(d)-1):
        min_ = min(pop, key=attrgetter('f1_'))
        max_ = max(pop, key=attrgetter('f1_'))
        f1 = pop[sortedf1[i+1]].f1_
        f2 = pop[sortedf2[i-1]].f2_
        d[i] = d[i]+(f1 - f2)/(max_.f1_-min_.f1_)

    for i in range(1, len(d)-1):
        min_ = min(pop, key=attrgetter('f2_'))
        max_ = max(pop, key=attrgetter('f2_'))
        f1 = pop[sortedf1[i+1]].f1_
        f2 = pop[sortedf2[i-1]].f2_
        d[i] = d[i]+(f1 - f2)/(max_.f2_-min_.f2_)
    return d


def crossover(sol1, sol2):
    """Function that realizes the aritmetical mean to crossover two Real values
    
    Arguments:
        sol1 {[float]} -- [Solution for the functions]
        sol2 {[float]} -- [Another solution for the functions]
    
    Returns:
        [float] -- [Crossovered value, between the two fathers]
    """    
    newX = (sol1.x_ + sol2.x_)/2
    newSol = Sol(newX)
    return newSol


def mutate(sol):
    """Function that based in a probability mutates the valuein two different values
    
    Arguments:
        sol {[float]} -- [Solution for the functions]
    
    Returns:
        [float] -- [The mutated solution]
    """    
    global mutateRate, minimum, maximum
    r = np.random.uniform()
    if r < mutateRate:
        r1 = np.random.uniform(0,2)
        if r1 < 0.5:
            sol.x_ = minimum+(maximum-minimum)*r
        else:
            sol.x_ = sol.x_*r1
    return sol


def createMutatedOffspring(pop):
    """Function that creates the offspring and mutates it
    
    Arguments:
        pop {[list]} -- [List of solutions]
    
    Returns:
        [list] -- [New list of solutions]
    """    
    newPop = []
    for i in range(len(pop)):
        i1 = random.randint(0, len(pop)-1)
        i2 = random.randint(0, len(pop)-1)
        newPop.append(crossover(pop[i1], pop[i2]))
    newPop = mutateOffspring(newPop)
    return newPop


def mutateOffspring(pop):
    """Function that mutates all the population
    
    Arguments:
        pop {[list]} -- [List of solutions to mutate]
    
    Returns:
        [list] -- [List of solutions mutated]
    """    
    for i in range(len(pop)):
        pop[i] = mutate(pop[i])
    return pop


def runNSGAExperiment(popSize, nGens):
    """Function that runs the NSGA II experiment
    
    Arguments:
        popSize {[int]} -- [Size of the population]
        numGen {[int]} -- [Number of generations]

    """ 
    pop1 = initPop(popSize)
    for j in range(nGens):
        n1 = [0] * len(pop1)
        s1 = [[] for i in range(len(pop1))]
        q1 = fastNonDominated(pop1, n1, s1)
        crowdingDistance1 = []
        for i in range(0, len(q1)):
            crowdingDistance1.append(getCrowding(pop1, n1, s1, q1[i]))

        newPop = createMutatedOffspring(pop1)
        pop2 = pop1 + newPop

        n2 = [0] * len(pop2)
        s2 = [[] for i in range(len(pop2))]
        q2 = fastNonDominated(pop2, n2, s2)
        crowdingDistance2 = []
        for i in range(0, len(q1)):
            crowdingDistance2.append(getCrowding(pop2, n2, s2, q2[i]))
        pop2Index = []
        pop3 = []
        for i in range(len(q2)):
            for j in range(len(q2[i])):
                pop3.append(q2[i][j])
                if len(pop3) == popSize:
                    break
            if len(pop3) == popSize:
                break
        pop1 = [pop2[i] for i in pop3]
 
    print("Final: \n")
    print("Index\t\t\tX_val\t\t\tF1\t\t\tF2")
    for j,o in enumerate(pop1):
        print(j, "\t\t\t",round(o.x_,1),"\t\t\t", o.f1_,"\t\t\t", o.f2_)
    print("\n")
    f1 = [i.f1_ for i in pop1]
    f2 = [i.f2_ for i in pop1]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.axhline(-1.0, color='r')
    plt.axvline(-1.0, color='r')
    plt.scatter(f1, f2)
    plt.show()


@click.command()
@click.option('--popsize', "-p", default=None, required=True,
              help=u'Size of the population.')
@click.option('--ngens', "-n", default=900, required=False,
              help=u'Size of the population.')
@click.option('--mini', "-m", default=-55, required=False,
              help=u'Size of the population.')
@click.option('--maxi', "-b", default=55, required=False,
              help=u'Size of the population.')
@click.option('--mutaterate', "-r", default=0.01, required=False,
              help=u'Size of the population.')
def main(popsize, ngens, mini, maxi, mutaterate):
    global minimum, maximum, mutateRate
    popSize = int(popsize)
    nGens = int(ngens)
    minimum = int(mini)
    maximum = int(maxi)
    mutateRate = float(mutaterate)
    runNSGAExperiment(popSize, nGens)


if __name__ == "__main__":
    main()
