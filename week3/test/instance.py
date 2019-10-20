import pandas as pd, numpy as np
import city
class Instance:
    
    def __init__(self):
        self.nCities = 0
        self.adjacencyMatrix = []
        self.listOfCities = []
        
    def readInstance(self, fileName):
        data = pd.read_csv(fileName, header=None)
        lol = data.values.tolist()
        for val in lol:
            val_city = city.City(val[0],val[1])
            self.listOfCities.append(val_city)
        self.nCities = len(lol)
        self.adjacencyMatrix = np.zeros((self.nCities, self.nCities))
        for i in range(self.nCities):
            for j in range(self.nCities):
                if i != j:
                    self.adjacencyMatrix[i][j]=self.listOfCities[i].euclideanDistance(self.listOfCities[j])
        print(self.adjacencyMatrix)
        
    def getDistance(city_1, city_2):
        return self.adjacencyMatrix[city_1][city_2]
    
    def getNNodes(self):
        return self.nCities