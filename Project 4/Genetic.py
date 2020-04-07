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

def crossover(pc):
    if random.random() <= 0.6: return 1
    else: return 0

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

def mutate(offspring, pm):
    for i in range(len(offspring)):
        if random.random() < pm:
            if offspring[i] == 0:
                offspring[i] = 1
            else:
                offspring[i] = 0
    
    return offspring


def calculateFitness(poparray, l, n, pm, pc):

    sumarray = []
    fitnessarray = []
    normalizedfitnessarray = []
    steppingtotal = []
    totalfitness = 0

    for i in range(len(poparray)):
        sumarray[i] = 0
        sumarray[i] = calculateRunningSum(sumarray[i], poparray[i])

        fitnessarray[i] = 0
        fitnessarray[i] = fitnessFunction(sumarray[i], l)

        totalfitness += fitnessarray[i]
    
    for i in range(len(poparray)):
        normalizedfitnessarray[i] = fitnessarray[i] / totalfitness
        if i:
            steppingtotal[i] += normalizedfitnessarray[i] + steppingtotal[i - 1]
        else:
            steppingtotal[i] += normalizedfitnessarray[i]
    
    rand1 = random.random()
    rand2 = random.random()

    parent1 = []
    parent2 = []
    offspring1 = []
    offspring2 = []

    newpoparray = []
    newpoparrayindex = 0

    for i in range(math.floor(n/2)):
        rand1bool = False
        rand2bool = False

        parent1index = 0
        parent2index = 0

        crossoverpoint = 0

        for j in range(len(steppingtotal)):
            if rand1 > steppingtotal[j]:
                parent1index = j + 1
                rand1bool = True 
                break

        

        while not rand1bool:
            for j in range(len(steppingtotal)):
                if rand2 > steppingtotal[j]:
                    if (j + 1 == parent1index):
                        rand2 = random.random()
                        break
                    parent2index = j + 1
                    rand2bool = True

        if crossover(pc):

            crossoverpoint = random.randint(0, l)

            for j in range(l):

                if(j <= crossoverpoint):
                    offspring1[j] = poparray[parent1index][j]
                    offspring2[j] = poparray[parent2index][j]
                else:
                    offspring2[j] = poparray[parent1index][j]
                    offspring1[j] = poparray[parent2index][j]

        else:
            offspring1 = poparray[parent1index]
            offspring2 = poparray[parent2index]

        offspring1 = mutate(offspring1, pm)
        offspring2 = mutate(offspring2, pm)

        newpoparray[newpoparrayindex] = offspring1
        newpoparrayindex += 1
        newpoparray[newpoparrayindex] = offspring2
        newpoparrayindex += 1

    averagefitness = totalfitness / n

    bestfitness = 0

    recorder = 0

    for i in range(len(fitnessarray)):
        if fitnessarray[i] > bestfitness:
            recorder = i
            bestfitness = fitnessarray[i]

    correctbits = 0

    for i in range(len(poparray[recorder])):
        if (poparray[recorder][i] == 1):
            correctbits += 1

    

    return newpoparray, averagefitness, bestfitness, correctbits

        
        
   
        
  


def algorithm(l, n, g, pm, pc):

    poparray = initialize()

    for G in range(g):
        poparray = calculateFitness(poparray, l, n, pm, pc)






        
        







