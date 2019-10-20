import numpy as np, random, operator, pandas as pd
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def euclideanDistance(self, city):
        x = abs(self.x - city.x)**2
        y = abs(self.y - city.y)**2
        return np.sqrt(x+y)