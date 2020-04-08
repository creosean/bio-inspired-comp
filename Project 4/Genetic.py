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
    poparray = [[0 for x in range(l)] for t in range(n)]

    for i in range(n):
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

    sumarray = [float] * n
    fitnessarray = [float] * n
    normalizedfitnessarray = [float] * n
    steppingtotal = [float] * n
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
            steppingtotal[i] = normalizedfitnessarray[i] + steppingtotal[i - 1]
        else:
            steppingtotal[i] =  normalizedfitnessarray[i]

    offspring1 = [int] * l
    offspring2 = [int] * l

    newpoparray = [[0 for x in range(l)] for t in range(n)]
    newpoparrayindex = 0


    for i in range(math.floor(n/2)):
        rand1bool = False
        rand2bool = False

        parent1index = 0
        parent2index = 0

        crossoverpoint = 0

        rand1 = random.random()
        rand2 = random.random()

        for j in range(len(steppingtotal)):
            if (rand1 < steppingtotal[j]):
                parent1index = j
                rand1bool = True 
                break
            


        while not rand1bool:
            for j in range(len(steppingtotal)):
                if (rand2 < steppingtotal[j]):
                    if (j == parent1index):
                        rand2 = random.random()
                        break
                    else:
                        parent2index = j
                        rand2bool = True
                        break


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



def algorithm(l, n, g, pm, pc, epochn, testn, csv_writer):

    poparray = initialize(l, n)
    poparray = initialize(l, n)

    averagefitnessarray = [float] * g
    bestfitnessarray = [float] * g
    correctbitsarray = [float] * g
    
    for G in range(g):
        poparray, averagefitnessarray[G], bestfitnessarray[G], correctbitsarray[G] = calculateFitness(poparray, l, n, pm, pc)

   
    


    csv_writer.writerow(["epoch number", "test number", "l", "N", "G", "Pm", "Pc"])
    csv_writer.writerow([epochn + 1, testn + 1, l, n, g, pm, pc])
    csv_writer.writerow(["averagefitness"])
    csv_writer.writerow(averagefitnessarray)
    csv_writer.writerow(["bestfitness"])
    csv_writer.writerow(bestfitnessarray)
    csv_writer.writerow(["numcorrectbits"])
    csv_writer.writerow(correctbitsarray)
    csv_writer.writerow([" "])
    


def testcycle(l, n, g, pm, pc, epochn, csv_writer):

    for i in range(3):
        algorithm(l, n, g, pm, pc, epochn, i, csv_writer)



l = []
l = [20, 30, 10, 50, 50, 50]

n = []
n = [30, 20, 40, 50, 50, 50]

pm = []
pm = [0.033, 0.1, 0.01, 0.5, 0.033, 0.02]

pc = []
pc = [0.6, 0.3, 0.9, 0.5, 0.6, 0.4]

g = []
g = [10, 25, 5, 50, 50, 50]

with open('data.csv', 'wt') as f:
    csv_writer = csv.writer(f)

    for i in range(len(g)):
        testcycle(l[i], n[i], g[i], pm[i], pc[i], i, csv_writer)
        csv_writer.writerow([" "])
        csv_writer.writerow([" "])



        






        
        







