##Producing Training/Testing inputs+output
from numpy import array, sin, cos, pi, tanh
from random import random

#Random initial angles
angle1 = random()
angle2 = random()

#The total 2*pi cycle would be divided into 'frequency'
#number of steps
frequency = 25
#This defines how many steps ahead we are trying to predict
lag = 23

# from numpy import sin,cos,pi,linspace
P=25
delta=-1

def get_sample():
    """
    Returns a [[sin value, cos value]] input.
    """
    global angle1, angle2
    angle1 += 2*pi/float(frequency)
    angle2 += 2*pi/float(frequency)
    angle1 %= 2*pi
    angle2 %= 2*pi

    # return array([array([
    #     5 + 5*sin(angle1) + 10*cos(angle2),
    #     7 + 7*sin(angle2) + 14*cos(angle1)])])
    return array([array([
        -4 * cos(angle1-1),
        -4 * cos(2*angle1-1)])])

    # return array([array([
    #     -0.4 * cos(2 * pi + delta),
    #     0.8 * sin(2 * pi)])])




sliding_window = []

for i in range(lag - 1):
    sliding_window.append(get_sample())


def get_pair():
    """
    Returns an (current, later) pair, where 'later' is 'lag'
    steps ahead of the 'current' on the wave(s) as defined by the
    frequency.
    """

    global sliding_window
    sliding_window.append(get_sample())
    input_value = sliding_window[0]
    output_value = sliding_window[-1]
    sliding_window = sliding_window[1:]
    return input_value, output_value
#Input Params
input_dim = 2

#To maintain state
last_value = array([0 for i in range(input_dim)])
last_derivative = array([0 for i in range(input_dim)])

def get_total_input_output():
    """
    Returns the overall Input and Output as required by the model.
    The input is a concatenation of the wave values, their first and
    second derivatives.
    """
    global last_value, last_derivative
    raw_i, raw_o = get_pair()
    raw_i = raw_i[0]
    l1 = list(raw_i)
    derivative = raw_i - last_value
    l2 = list(derivative)
    last_value = raw_i
    l3 = list(derivative - last_derivative)
    last_derivative = derivative

    from operator import add
    temp = map(add, l1, l2)
    temp = map(add, temp, l3)
    return array([temp]), raw_o


# actual_output1 = []
# actual_output2 = []
# x_axis = []
# frequency1=1000
# import numpy as np
# gauss_n = np.zeros((2, frequency1));
# gauss_n[0] = np.random.normal(0, 0.1, frequency1)
# gauss_n[1] = np.random.normal(0, 0.3, frequency1)
#
# for i in range(1000):
#     input_v, output_v = get_total_input_output()
#
#     actual_output1.append(output_v[0][0] + gauss_n[0][i])
#     actual_output2.append(output_v[0][1] + gauss_n[1][i])
#     x_axis.append(i)
#
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(actual_output1, actual_output2)
# plt.show()


actual_output1 = []
actual_output2 = []
network_output1 = []
network_output2 = []
x_axis = []
error_temp = []
error_train = []
error_test1 = []
error_test2 = []
liss_x=[]
liss_y=[]
liss_x_ac=[]
liss_y_ac=[]

# input_v, output_v = get_total_input_output()
# print input_v
# print output_v
iter = 100000
import numpy as np

lissajous_period = 1000
gauss_n = np.zeros((2, lissajous_period));
gauss_n[0] = np.random.normal(0, 0.1, lissajous_period)
gauss_n[1] = np.random.normal(0, 0.3, lissajous_period)

for i in range(iter):
    input_v, output_v = get_total_input_output()

    input_v[0][0] = input_v[0][0] + gauss_n[0][np.mod(i, lissajous_period)]
    input_v[0][1] = input_v[0][1] + gauss_n[1][np.mod(i, lissajous_period)]
    output_v[0][0] = output_v[0][0] + gauss_n[0][np.mod(i, lissajous_period)]
    output_v[0][1] = output_v[0][1] + gauss_n[1][np.mod(i, lissajous_period)]

    actual_output1.append(output_v[0][0])
    actual_output2.append(output_v[0][1])
    network_output1.append(input_v[0][0])
    network_output2.append(input_v[0][1])
    x_axis.append(i)

    if ((np.mod (i, lissajous_period)==0) and (i != 0)):
        error_x = np.square(np.subtract(liss_x, liss_x_ac))
        error_y = np.square(np.subtract(liss_y, liss_y_ac))

        error_sumx = np.sum(error_x)
        error_sumy = np.sum(error_y)
        error_temp = (error_sumx + error_sumy) / (input_dim * lissajous_period)

        error_train.append(error_temp)

        liss_x = []
        liss_y = []
        liss_x_ac = []
        liss_y_ac = []

    liss_x.append(input_v[0][0])
    liss_y.append(input_v[0][1])

    liss_x_ac.append(output_v[0][0])
    liss_y_ac.append(output_v[0][1])

error_x = np.square(np.subtract(liss_x, liss_x_ac))
error_y = np.square(np.subtract(liss_y, liss_y_ac))

error_sumx = np.sum(error_x)
error_sumy = np.sum(error_y)
error_temp = (error_sumx + error_sumy) / (input_dim * lissajous_period)

error_train.append(error_temp)

print error_train
print len(error_train)

import matplotlib.pyplot as plt
t_axis = np.arange(0,iter/lissajous_period,1)
plt.figure(1)
plt.plot(t_axis, error_train, 'r-')
plt.show()












# import json
#
# angle1 = random()
# angle2 = random()
#
# #The total 2*pi cycle would be divided into 'frequency'
# #number of steps
# frequency1 = 25
# frequency2 = 25
#
#
# angle1 += 2 * pi / float(frequency1)
# angle2 += 2 * pi / float(frequency2)
# angle1 %= 2 * pi
# angle2 %= 2 * pi
# # return array([array([
# #     5 + 5*sin(angle1) + 10*cos(angle2),
# #     7 + 7*sin(angle2) + 14*cos(angle1)])])
# t = np.arange(0, 100000, 1)
# print (array([array([
#     -4 * cos(angle1 * t - 1),
#     -4 * cos(2 * angle1 * t - 1)])]))
# # f = open('workfile.txt', 'r+')
# # # f.write('(array([array([-4 * cos(angle1 * t - 1),-4 * cos(2 * angle1 * t - 1)])]))')
# # json.dumps((array([array([-4 * cos(angle1 * t - 1),-4 * cos(2 * angle1 * t - 1)])])),f)
