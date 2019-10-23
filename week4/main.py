import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import time
import sys
import itertools


def getAllPerms(n):
    route = np.arange(n)
    perm(route, 0, n)


def perm(route, start, ending):
    for i in range(start, ending):
        route[start], route[i] = route[i], route[start]
        perm(route, start+1, ending)
        route[start], route[i] = route[i], route[start]


def plotResults(time, cities):
    plt.figure()
    axes = plt.axes()
    axes.plot(cities, time)
    plt.xlabel('N Cities')
    plt.ylabel('Time (s)')
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Incorrect number of parameters: \n" +
              "\t main.py \"min\" \"max\"")
        return -1

    min = int(sys.argv[1])
    max = int(sys.argv[2])+1
    time_elapsed = []
    for i in range(min, max):
        time.process_time()
        getAllPerms(i)
        time_elapsed.append(time.process_time())
    count = min
    cities = []
    print("Time \t N_cities")
    for index, val in enumerate(time_elapsed):
        print(val, "\t", count)
        cities.append(count)
        count += 1

    plotResults(time_elapsed, cities)


if __name__ == "__main__":
    main()
