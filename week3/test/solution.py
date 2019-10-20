from evaluate import Evaluator
import numpy as np
class Solution:
    def __init__(self, n_nodes):
        self.route = np.zeros(n_nodes)
        self.fitness = 0
        
    def putCityIn(self, pos, city):
        self.route[pos] = city
        
    def setFitness(self, fitness):
        self.fitness = fitness