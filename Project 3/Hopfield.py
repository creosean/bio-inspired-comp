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

def initialize_array():
    Hnet = np.zeros((50, 100))

    for i in range(50):
        for j in range(100):
            Hnet[i][j] = twoRandomNumbers(1, -1)

    return Hnet

def display_result_terminal(Hnet):
    for i in range(50):
        for j in range(100):
            if(Hnet[i][j] == 1):
                print('1', end=' ')
            else: print('0', end=' ')
        print(end='\n')

def find_weights(p, hnet_array):
    weight_array = np.zeros((100,)*2)
    weight = 0
    
    for i in range(100):
        for j in range(100):
            if(i == j):
                weight_array[i][j] = 0 
                continue
            else:
                weight = 0
                for k in range(p):
                    weight += hnet_array[i] * hnet_array[j]
                if(weight != 0):
                    weight /= 100
                
                weight_array[i][j] = weight
    
    return weight_array

def compute_state_value(hnet_array, weight_array):
    new_state = np.zeros(100)

    state_value = 0

    for i in range(100):
        state_value = 0
        for j in range(100):
            state_value += weight_array[i][j] * hnet_array[j]
        
        new_state[i] = state_value
    
    for i in range(100):
        if(new_state[i] >= 0):
            new_state[i] = 1
        else:
            new_state[i] = -1
    
    return new_state

def is_stable(hnet_array, new_state):

    counter = 0

    for i in range(100):
        if(hnet_array[i] != new_state[i]):
            return 1
    return 0



stable_probility = np.zeros(50)

unstable_probability = np.zeros(50)

stables = np.zeros(50)

for k in range(5):

    Hnet = initialize_array()

    display_result_terminal(Hnet)

    stable = 0
    unstable = 0

    stability = np.zeros(50)

    for p in range(50):

        weight_array = find_weights(p, Hnet[p])

        new_state = compute_state_value(Hnet[p], weight_array)

        if(not (is_stable(Hnet[p], new_state))):
            stability[p] += 1
            stables[p] += 1

    for p in range(50):

        if(stability[p] != 0 and (stability[p] / (p + 1)) != 0 ):
            stable_probility[p] += (stability[p] / (p + 1)) / 5

        if((1 - stable_probility[p]) != 0):
            unstable_probability[p] += (1 - stable_probility[p])/5
            

with open('data.csv', 'wt') as f:
    csv_writer = csv.writer(f)

    csv_writer.writerow(stable_probility)
    csv_writer.writerow(unstable_probability)
    csv_writer.writerow(stables)




