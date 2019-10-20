import randomSol
from solution import Solution
from instance import Instance
from evaluate import Evaluator
class GA:
    def __init__(self, n_nodes):
        self.popSize = 0
        self.population = []
        self._results = []
        self._popMeanResults = []
        self._offMeanResults = []
        self._bestPerIteration = []
        self._bestSolution = Solution(n_nodes)
        self._bestSolution.setFitness(0)
    
    def initPopulation(self, popSize, n_nodes, inst = Instance):
        self.popSize = popSize
        
        fitness = 0
        firstIt = True
        ev = Evaluator()
        for i in range(self.popSize):
            sol = randomSol.generateRandomSolution(n_nodes)
            print(sol.route)
            ev.setRoute(sol.route)
            #print(str(sol.getRoute())+'\t'+str(ev.route)+'\t'+str(i)+'\n')
            fitness = ev.computeFitness(inst)
            if (firstIt) == True or (fitness-self._bestSolution.getFitness())>0:
                firstIt = False
                self._bestSolution = sol
            self._results.append(fitness)
            self.population.append(sol)
                 