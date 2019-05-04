##Producing Training/Testing inputs+output
from numpy import array, sin, cos, pi
from random import random
 
#Random initial angles
angle1 = random()
angle2 = random()
 
#The total 2*pi cycle would be divided into 'frequency'
#number of steps
frequency1 = 200
frequency2 = 200
#This defines how many steps ahead we are trying to predict
lag = 15
 
def get_sample():
    """
    Returns a [[sin value, cos value]] input.
    """
    global angle1, angle2
    angle1 += 2*pi/float(frequency1)
    angle2 += 2*pi/float(frequency2)
    angle1 %= 2*pi
    angle2 %= 2*pi
    return array([array([
        5 + 5*sin(angle1) + 10*cos(angle2),
        7 + 7*sin(angle2) + 14*cos(angle1)])])
 
 
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
    # return array([l1 + l2 + l3]), raw_o
#Imports
import tensorflow as tf
from tensorflow.models.rnn.rnn import *

#Input Params
input_dim = 2


##The Input Layer as a Placeholder
#Since we will provide data sequentially, the 'batch size'
#is 1.
input_layer = tf.placeholder(tf.float32, [1, input_dim])

# ##First Order Derivative Layer
# #This will store the last recorded value
# last_value1 = tf.Variable(tf.zeros([1, input_dim]))
# #Subtract last value from current
# sub_value1 = tf.sub(input_layer, last_value1)
# #Update last recorded value
# last_assign_op1 = last_value1.assign(input_layer)
#
# ##Second Order Derivative Layer
# #This will store the last recorded derivative
# last_value2 = tf.Variable(tf.zeros([1, input_dim]))
# #Subtract last value from current
# sub_value2 = tf.sub(sub_value1, last_value2)
# #Update last recorded value
# last_assign_op2 = last_value2.assign(sub_value1)
#
# ##Overall input to the LSTM
# #x and its first and second order derivatives as outputs of
# #earlier layers
# zero_order = last_assign_op1
# first_order = last_assign_op2
# second_order = sub_value2
# #Concatenated
# total_input = tf.concat(1, [zero_order, first_order, second_order])
#######################




##The LSTM Layer-1
#The LSTM Cell initialization
lstm_layer1 = rnn_cell.BasicLSTMCell(input_dim*3)
#The LSTM state as a Variable initialized to zeroes
lstm_state1 = tf.Variable(tf.zeros([1, lstm_layer1.state_size]))
#Connect the input layer and initial LSTM state to the LSTM cell
lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1)
                                               # ,scope=&quot,LSTM1&quot)
#The LSTM state will get updated
lstm_update_op1 = lstm_state1.assign(lstm_state_output1)


##The Regression-Output Layer1
#The Weights and Biases matrices first
output_W1 = tf.Variable(tf.truncated_normal([input_dim*3, input_dim]))
output_b1 = tf.Variable(tf.zeros([input_dim]))
#Compute the output
final_output = tf.matmul(lstm_output1, output_W1) + output_b1

##Input for correct output (for training)
correct_output = tf.placeholder(tf.float32, [1, input_dim])

##Calculate the Sum-of-Squares Error
error = tf.pow(tf.sub(final_output, correct_output), 2)

##The Optimizer
#Adam works best
train_step = tf.train.AdamOptimizer(0.0006).minimize(error)

##Session
sess = tf.Session()
#Initialize all Variables
sess.run(tf.initialize_all_variables())

##Training

actual_output1 = []
actual_output2 = []
network_output1 = []
network_output2 = []
x_axis = []
error_train1 = []
error_train2 = []
error_test1 = []
error_test2 = []
liss_x=[]
liss_y=[]

# input_v, output_v = get_total_input_output()
# print input_v
# print output_v
iter = 100000
import numpy as np
for i in range(iter):
    input_v, output_v = get_total_input_output()
    _, _, network_output = sess.run([lstm_update_op1,
                                     train_step,
                                     final_output],
                                    feed_dict = {
                                        input_layer: input_v,
                                        correct_output: output_v})

    actual_output1.append(output_v[0][0])
    actual_output2.append(output_v[0][1])
    network_output1.append(network_output[0][0])
    network_output2.append(network_output[0][1])
    x_axis.append(i)

    error_temp = np.square(network_output[0][0] - output_v[0][0])
    error_train1.append(error_temp)

    error_temp = np.square(network_output[0][1] - output_v[0][1])
    error_train2.append(error_temp)

    if (i >= iter - frequency1 + 1):
        liss_x.append(network_output[0][0])
        liss_y.append(network_output[0][1])

import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(x_axis, network_output1, 'r-', x_axis, actual_output1, 'b-')
# plt.show()
# plt.plot(x_axis, network_output2, 'r-', x_axis, actual_output2, 'b-')
# plt.show()
#
# plt.figure(2)
# plt.plot(x_axis, error_train1, 'b-')
# plt.show()
# plt.plot(x_axis, error_train2, 'b-')
# plt.show()

plt.figure(1)
plt.plot(liss_x, liss_y, 'b-')
plt.show()
plt.figure(2)
plt.plot(actual_output1, actual_output2, 'r-')
plt.show()

##Testing

for i in range(200):
    get_total_input_output()

#Flush LSTM state
sess.run(lstm_state1.assign(tf.zeros([1, lstm_layer1.state_size])))
actual_output1 = []
actual_output2 = []
network_output1 = []
network_output2 = []
x_axis = []


for i in range(1000):
    input_v, output_v = get_total_input_output()
    _, network_output = sess.run([lstm_update_op1,
                                  final_output],
                                 feed_dict = {
                                     input_layer: input_v,
                                     correct_output: output_v})

    actual_output1.append(output_v[0][0])
    actual_output2.append(output_v[0][1])
    network_output1.append(network_output[0][0])
    network_output2.append(network_output[0][1])
    x_axis.append(i)

    error_temp = np.square(network_output[0][0] - output_v[0][0])
    error_test1.append(error_temp)

    error_temp = np.square(network_output[0][1] - output_v[0][1])
    error_test2.append(error_temp)


plt.figure(3)
plt.plot(x_axis, network_output1, 'r-', x_axis, actual_output1, 'b-')
plt.show()
plt.plot(x_axis, network_output2, 'r-', x_axis, actual_output2, 'b-')
plt.show()

plt.figure(4)
plt.plot(x_axis, error_test1, 'b-')
plt.show()
plt.plot(x_axis, error_test2, 'b-')
plt.show()