import city
from instance import Instance
class Evaluator:
    def __init__(self):
        self.route = []
        self.distance = 0
        self.fitness = 0

    def setRoute(self, route):
        self.route = route
    
    def getRouteDistance(self, inst=Instance):
        if self.distance == 0:
            for i in range(len(self.route)):
                fromCity = i
                toCity = None
                if (i + 1) < len(self.route):
                    toCity = i+1
                else:
                    toCity = 0
                self.distance+= inst.getDistance(fromCity, toCity)
        return self.distance
       
    def computeFitness(self, inst=Instance):
        if self.fitness == 0:
            self.fitness = 1/float(self.getRouteDistance(inst))
        return fitness