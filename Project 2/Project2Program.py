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

def write_to_pgm(array_2d, name):
    f = open(name,"w+")

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

#The notation here is a bit convoluted, but I am keeping it as such for continuity between this and the intructions
#i refers to a cell, and i1 would refer to its first index, and i2 the second. Intuitively, i1 would be my i coordinate, and i2 the j coordinate.
def calculate_distance(i1, i2, j1, j2):
    x = abs(i1 - j1)
    if(x > 15): x = 30 - x

    y = abs(i2 - j2)
    if(y > 15): y = 30 - y

    return (x + y)


def initialize_experiment(experiment):
    
    h = np.array([])
    R2 = np.array([])
    R1 = np.array([])
    j = np.array([])
    
    if(experiment == 1):
        h = [-1, -1, -2]
        R2 = [15, 15, 15]
        R1 = [1, 3, 6]
        j = [1, 0]

    elif(experiment == 2):
        h = [0, -2, -1, 0, -5, -3, 0, 0, 0, 0, -5, -3, 0, 0, -6, -3, 0]
        R2 = [2, 4, 4, 4, 6, 6, 6, 9, 13, 5, 7, 7, 7, 12, 12, 12, 12]
        R1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 9, 9, 9]
        j = [0, -0.1]

        
    elif(experiment == 3):
        h = [0, -4, -2, 0, -6, -3, 0, 0, -1, 0, -6, -3, 0, 0, 0, 0, 0]
        R2 = [2, 5, 5, 5, 9, 9, 9, 14, 5, 5, 9, 9, 9, 14, 9, 14, 14]
        R1 = [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 7, 7, 12]
        j = [1, -0.1]

    
    return np.array([R1, R2, h, j])



#array_2d = initialize_array()

#display_result_terminal(array_2d)

#write_to_pgm(array_2d)

experiment_list = initialize_experiment(3)

current_experiment = "experiment_one"

number_flipped_cells = 0
number_iterations = 0

with open('data.csv', 'wt') as f:
    csv_writer = csv.writer(f)

    for e in range(3):

        experiment_list = initialize_experiment(e + 1)

        R1 = experiment_list[0]
        R2 = experiment_list[1]
        H = experiment_list[2]
        j1 = experiment_list[3][0]
        j2 = experiment_list[3][1]

        if(e == 1):
            current_experiment = "experiment_two"
        elif(e == 2): current_experiment = "experiment_three"

        for p in range(len(R1)):

            r1 = R1[p]
            r2 = R2[p]
            h = H[p]

            

            array_2d = initialize_array()

            correlation = np.zeros(15)
            joint_entropy = np.zeros(15)
            mutual_information = np.zeros(15)

            while(1):
                number_flipped_cells = 0
                #this creates a list of every possible cell, and then shuffles the order. It is the cartesian product, so we dont need to use the 30x30 array and search through it every time
                all_combinations = list(itertools.product([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], repeat=2))

                shuffled_indices = np.arange(900)
                np.random.shuffle(shuffled_indices)

                near_cells = 0
                far_cells = 0

                for n in range(len(all_combinations)):

                    current_index = shuffled_indices[n]
                    i1 = all_combinations[current_index][0]
                    i2 = all_combinations[current_index][1]

                    past_value = array_2d[i1][i2]

                    for x in range(30):
                        for y in range(30):
                            distance = calculate_distance(i1, i2, x, y)
                            if(distance < r1): near_cells += array_2d[x][y]
                            elif(distance < r2): far_cells += array_2d[x][y]

                    near_cells *= j1
                    far_cells *= j2

                    if((h + far_cells + near_cells) > 1): 
                        array_2d[i1][i2] = 1
                        if(past_value != 1): number_flipped_cells += 1
                    else:
                        array_2d[i1][i2] = -1
                        if(past_value != -1): number_flipped_cells += 1

                
                #display_result_terminal(array_2d)

                number_iterations += 1
                if(number_flipped_cells == 0 or number_iterations > 30): break
            
        

            for l in range(15):

                sum_i = 0
                sum_ij = 0

                for xi in range(30):
                    for yi in range(30):
                        sum_i += array_2d[xi][yi]

                        for xj in range(30):
                            #if(xj < xi): continue
                            for yj in range(30):
                                if((xi == xj) and (yj <= yi)):
                                    continue
                                elif(l == calculate_distance(xi, yi, xj, yj)):
                                    sum_ij += array_2d[xi][yi] * array_2d[xj][yj]
                if(l > 0):
                    correlation[l] = abs( ((2/(30 * 30 * 4 * l)) * sum_ij) - ((1/(30 * 30)) * sum_i)**2 )        
                else: correlation[l] = abs( 1 - ((1/(30 * 30)) * sum_i)**2 )

            
            sum_converted = 0

            for x in range(30):
                for y in range(30):
                    sum_converted += (1 + array_2d[x][y]) / 2

            prob_pos = (1 / (30 * 30)) * sum_converted
            prob_neg = 1 - prob_pos

            overall_entropy = 0

            if(prob_neg == 0 and prob_pos == 0):
                overall_entropy = 0

            elif(prob_neg == 0 and prob_pos != 0):
                overall_entropy = -(prob_pos * math.log(prob_pos))

            elif(prob_neg != 0 and prob_pos == 0):
                overall_entropy = -(prob_neg * math.log(prob_neg))

            else:
                overall_entropy = -((prob_pos * math.log(prob_pos)) + (prob_neg * math.log(prob_neg)))


            

            for l in range(15):
                sum_pos = 0
                sum_neg = 0
                pos_entropy = 0
                neg_entropy = 0
                fuzzy_entropy = 0

                for xi in range(30):
                    for yi in range(30):
                        #if(xj < xi): continue
                        for xj in range(30):
                            for yj in range(30):
                                if((xi == xj) and (yj <= yi)):
                                    continue
                                elif(l == calculate_distance(xi, yi, xj, yj)):
                                    sum_pos += ((1 + array_2d[xi][yi]) / 2) * ((1 + array_2d[xj][yj]) / 2)
                                    sum_neg += (((1 - array_2d[xi][yi]) / 2)) * (((1 - array_2d[xj][yj]) / 2))
                if(l > 0):
                    pos_entropy = ((2/(30 * 30 * 4 * l)) * sum_pos) 
                    neg_entropy = ((2/(30 * 30 * 4 * l)) * sum_neg)        
                else: 
                    pos_entropy = (1) 
                    neg_entropy = (1)

                fuzzy_entropy = 1 - pos_entropy - neg_entropy

                if(pos_entropy == 0):
                    pos_entropy = 0
                else:
                    pos_entropy *= math.log(pos_entropy)

                if(neg_entropy == 0):
                    neg_entropy = 0
                else:
                    neg_entropy *= math.log(neg_entropy)

                fuzzy_entropy =  abs(fuzzy_entropy)
                if(fuzzy_entropy == 0):
                    fuzzy_entropy = 0
                else:
                    fuzzy_entropy *= math.log(fuzzy_entropy)

                joint_entropy[l] = abs(pos_entropy + neg_entropy + fuzzy_entropy)
            
            for l in range(15):
                mutual_information[l] = abs((2 * overall_entropy) - joint_entropy[l])

            

            csv_writer.writerow(['current experiment', 'number' 'r1', 'r2', 'h', 'j1', 'j2', 'overall entropy'])
            csv_writer.writerow([current_experiment, r1, r2, h, j1, j2, overall_entropy])
            csv_writer.writerow(correlation)
            csv_writer.writerow(joint_entropy)
            csv_writer.writerow(mutual_information)
                
            name = current_experiment + "_" + str(p + 1) + ".pgm"

            write_to_pgm(array_2d, name)
        




        
        







