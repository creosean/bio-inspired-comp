import random
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats

def twoRandomNumbers(a,b):
    if random.random() < 0.5: return a
    else: return b

def initialize_array():
    array_2d = np.zeros((30,)*2)

    for i in range(30):
        for j in range(30):
            array_2d[i][j] = twoRandomNumbers(1, -1)

    return array_2d


def display_result_terminal(array_2d):
    for i in range(30):
        for j in range(30):
            if(array_2d[i][j] == 1):
                print('*', end=' ')
            else: print(' ', end=' ')
        print(end='\n')

def write_to_pgm(array_2d):
    f = open("AICA_img.pgm","w+")

    f.write("P2\n")
    f.write("300 300\n")
    f.write("255\n")

    for i in range(30):
        for k in range(10):
            for j in range(30):
                for n in range(10):
                    if(array_2d[i][j] == 1):
                        f.write("0 ")
                    else: f.write("255 ")
                f.write('\n')

def calculate_distance(i1, j1, i2, j2):
    return (abs(i1 - j1) + abs(i2 - j2))

def initialize_experiment(experiment):
    
    h = np.array([])
    R2 = np.array([])
    R1 = np.array([])
    
    if(experiment == 1):
        np.append(h, [-1, -1, -2])
        np.append(R2, [15, 15, 15])
        np.append(R1, [1, 3, 6])

    elif(experiment == 2):
        np.append(h, [0, -2, -1, 0, -5, -3, 0, 0, 0, 0, -5, -3, 0, 0, -6, -3, 0])
        np.append(R2, [2, 4, 4, 4, 6, 6, 6, 9, 13, 5, 7, 7, 7, 12, 12, 12, 12])
        np.append(R1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 9, 9, 9])
        
    elif(experiment == 3):
        np.append(h, [0, -4, -2, 0, -6, -3, 0, 0, -1, 0, -6, -3, 0, 0, 0, 0, 0])
        np.append(R2, [2, 5, 5, 5, 9, 9, 9, 14, 5, 5, 9, 9, 9, 14, 9, 14, 14])
        np.append(R1, [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 7, 7, 12])
    
    return np.array([R1, R2, h])

R1 = np.array(15)

R2 = np.array(15)

h = np.array(15)

array_2d = initialize_array()

display_result_terminal(array_2d)

write_to_pgm(array_2d)

experiment_list = initialize_experiment(3)
