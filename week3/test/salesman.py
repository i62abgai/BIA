import randomSol
from geneticAlgorithm import GA
from evaluate import Evaluator
from instance import Instance
import sys, numpy as np

def main():
    if len(sys.argv) != 2:
        print("Incorrect number of parameters: \n"+
        "\t salesman.py \"instance file\"")
        return -1
    
    inst=Instance()
    inst.readInstance(sys.argv[1])
    
    ga = GA(inst.nCities)
    ga.initPopulation(10, inst.nCities, inst)

if __name__ == "__main__":
    main()