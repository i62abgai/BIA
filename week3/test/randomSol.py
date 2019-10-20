from solution import Solution
import numpy as np
def generateRandomSolution(n_nodes):
    perm = np.random.permutation(n_nodes)
    sol = Solution(n_nodes)
    for i, val in enumerate(perm):
        sol.putCityIn(i, val)
    return sol