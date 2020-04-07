import random
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import itertools
import csv

def twoRandomNumbers(a,b):
    if random.random() < 0.5: return a
    else: return b

def initialize(l, n):
    poparray = []

    for i in range(n):
        poparray[i] = []

        for j in range(l):
            poparray[i][j] = twoRandomNumbers(1,0)

    return poparray

def calculateRunningSum(runningsum, individual):

    for i in range(len(individual)):
        if individual[i] == 1:
            runningsum += pow(2, len(individual) - i - 1)
    
    return runningsum

def fitnessFunction(x, l):
    return ( (x / (2 ** l)) ** 10)

def calculateFitness(poparray, l):

    sumarray = []
    fitnessarray = []
    normalizedfitnessarray = []
    totalfitness = 0

    for i in range(len(poparray)):
        sumarray[i] = 0
        sumarray[i] = calculateRunningSum(sumarray[i], poparray[i])

        fitnessarray[i] = 0
        fitnessarray[i] = fitnessFunction(sumarray[i], l)

        totalfitness += fitnessarray[i]



def algorithm(l, n, g, pm, pc):

    poparray = initialize()




        
        







