import random
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import itertools
import csv

def update_velocity(inertia, velocity, c_1, r_1, personal_best_position, position, c_2, r_2, global_best_position):
    return inertia * velocity + c_1 * r_1 * (personal_best_position - position) + c_2 * r_2 * (global_best_position - position)

def affirm_max_velocity(velocity, v_x, v_y, max_v):
    if (v_x ** 2 + v_y ** 2 > max_v ** 2):
        velocity = (max_v / math.sqrt(v_x ** 2 + v_y ** 2)) * velocity
    return velocity

def update_position(position, velocity):
    return position + velocity

def absolute(pos_x, pos_y):
    return math.sqrt(pos_x ** 2 + pos_y ** 2)

def calc_error(k, pos_list, global_best):
    
    error = 0
    
    for i in range(k):
        error += (pos_list[i] - global_best) ** 2

    error = math.sqrt(( 1 / ( 2 * k)) * error)

    return error

def generate_rand_v(width, height):
    upper_h_bound = (height / 2)
    lower_h_bound = upper_h_bound - height
    upper_w_bound = (width / 2)
    lower_w_bound = upper_w_bound - width

    x_pos = random.uniform(lower_w_bound, upper_w_bound)
    y_pos = random.uniform(lower_h_bound, upper_h_bound)

    return x_pos, y_pos

def q1(pdist, mdist):
    return 100 * (1 - (pdist / mdist))

def q2(pdist, mdist, ndist):
    
    v1 = 0

    if(10 - pdist ** 2 > 0):
        vi = 9 * (10 - pdist ** 2)
    
    return v1 + (10 * (1 - (pdist / mdist))) + (70 * (1 - (ndist / mdist)))

def Q(x, y, mdist, pdist, ndist, p_num):
    if(p_num is 1):
        return q1(pdist, mdist)

    elif(p_num is 2):
        return q2(pdist, mdist, ndist)
    
    else:
        return absolute(x, y)


def initialize(num_particles, width, height, p_num):
    pos_x_list = np.array([])
    pos_y_list = np.array([])
    vel_x_list = np.array([])
    vel_y_list = np.array([])
    personalbest_x_list = np.array([])
    personalbest_y_list = np.array([])
    pdist = np.array([])
    ndist = np.array([])

    mdist = math.sqrt((height ** 2 + width ** 2)) / 2

    globalbest_x = 0
    globalbest_y = 0

    x = 0
    y = 0

    for i in range(num_particles):
        x, y = generate_rand_v(width, height)
        
        pos_x_list = np.append(pos_x_list, x)
        pos_y_list = np.append(pos_y_list, y)

        vel_x_list = np.append(vel_x_list, 0)
        vel_y_list = np.append(vel_y_list, 0)

        personalbest_x_list = np.append(personalbest_x_list, pos_x_list[i])
        personalbest_y_list = np.append(personalbest_y_list, pos_y_list[i])

        pdist = np.append(pdist, math.sqrt((pos_x_list[i] - 20) ** 2 + (pos_y_list[i] - 7) ** 2))
        ndist = np.append(ndist, math.sqrt((pos_x_list[i] + 20) ** 2 + (pos_y_list[i] + 7) ** 2))

        if(Q(personalbest_x_list[i], personalbest_y_list[i], mdist, pdist[i], ndist[i], p_num) > Q(globalbest_x, globalbest_y, mdist, pdist[i], ndist[i], p_num)):
            globalbest_x = personalbest_x_list[i]
            globalbest_y = personalbest_y_list[i]

    return np.array([pos_x_list, pos_y_list, vel_x_list, vel_y_list, personalbest_x_list, personalbest_y_list]), globalbest_x, globalbest_y, mdist, pdist, ndist

problem_num = 2

max_epochs = 10000
epochs_reached = 0

width = 100
height = 100

num_particles = 20

inertia = 1

max_v = 10

c_1 = 2
c_2 = 2

r_1 = 0
r_2 = 0

mdist = 0
pdist = np.array([])
ndist = np.array([])
error_x = 0
error_y = 0

x_error_list = np.array([])
y_error_list = np.array([])

globalbest_x = 0
globalbest_y = 0

pos_vel_pers, globalbest_x, globalbest_y, mdist, pdist, ndist = initialize(num_particles, width, height, problem_num)

for i in range(max_epochs):

    for n in range(num_particles):
        r_1 = random.uniform(0, 1)
        r_2 = random.uniform(0, 1)
        pos_vel_pers[2][n] = update_velocity(inertia, pos_vel_pers[2][n], c_1, r_1, pos_vel_pers[4][n], pos_vel_pers[0][n], c_2, r_2, globalbest_x)
        pos_vel_pers[3][n] = update_velocity(inertia, pos_vel_pers[3][n], c_1, r_1, pos_vel_pers[5][n], pos_vel_pers[1][n], c_2, r_2, globalbest_y)

        pos_vel_pers[2][n] = affirm_max_velocity(pos_vel_pers[2][n], pos_vel_pers[2][n], pos_vel_pers[3][n], max_v)
        pos_vel_pers[3][n] = affirm_max_velocity(pos_vel_pers[3][n], pos_vel_pers[2][n], pos_vel_pers[3][n], max_v)


        pos_vel_pers[0][n] = update_position(pos_vel_pers[0][n], pos_vel_pers[2][n])
        pos_vel_pers[1][n] = update_position(pos_vel_pers[1][n], pos_vel_pers[3][n])

        pdist[n] = math.sqrt((pos_vel_pers[0][n] - 20) ** 2 + (pos_vel_pers[1][n] - 7) ** 2)
        ndist[n] = math.sqrt((pos_vel_pers[0][n] + 20) ** 2 + (pos_vel_pers[1][n] + 7) ** 2)

        if(Q(pos_vel_pers[0][n], pos_vel_pers[1][n], mdist, pdist[n], ndist[n], problem_num) > Q(pos_vel_pers[4][n], pos_vel_pers[5][n], mdist, pdist[n], ndist[n], problem_num)):
            pos_vel_pers[4][n] = pos_vel_pers[0][n]
            pos_vel_pers[5][n] = pos_vel_pers[1][n]

            if(Q(pos_vel_pers[4][n], pos_vel_pers[5][n], mdist, pdist[n], ndist[n], problem_num) > absolute(globalbest_x, globalbest_y, mdist, pdist[n], ndist[n], problem_num)):
                globalbest_x = pos_vel_pers[4][n]
                globalbest_y = pos_vel_pers[5][n]
        




    error_x = calc_error(num_particles, pos_vel_pers[0], globalbest_x)
    error_y = calc_error(num_particles, pos_vel_pers[1], globalbest_y)

    epochs_reached += 1

    x_error_list = np.append(x_error_list, error_x)
    y_error_list = np.append(y_error_list, error_y)

    if(error_x < 0.01 and error_y < 0.01):
        break

    if(i == max_epochs / 1000):
        print(error_x)
        print(error_y)

print(epochs_reached)
print(error_x)
print(error_y)


plt.scatter(pos_vel_pers[0], pos_vel_pers[1], c="blue")
plt.ylim(top = height / 2)
plt.ylim(bottom = (height / 2) - height)
plt.xlim(right = width / 2)
plt.xlim(left = (width / 2) - width)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.show()
plt.scatter(pos_vel_pers[0], pos_vel_pers[1], c="blue")
plt.ylim(top = height / 2)
plt.ylim(bottom = (height / 2) - height)
plt.xlim(right = width / 2)
plt.xlim(left = (width / 2) - width)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
if(problem_num == 1):
    plt.scatter([20], [7], c="red", marker="^")
elif(problem_num == 2):
    plt.scatter([20, -20], [7, -7], c="red", marker="^")
plt.show()

