import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

import csv

##DATA LOADING AND RESHAPNG
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

print(type(x_train))
print(type(y_train))

print(str(len(x_train)))

trainIndexesToDelete = []
for i in range(0, len(y_train)):
    if (y_train[i] != 0) and (y_train[i] != 1) and (y_train[i] != 8) and (y_train[i] != 9):
        trainIndexesToDelete.append(i)

testIndexesToDelete = []
for i in range(0, len(y_test)):
    if (y_test[i] != 0) and (y_test[i] != 1) and (y_test[i] != 8) and (y_test[i] != 9):
        testIndexesToDelete.append(i)

print(trainIndexesToDelete)
print(testIndexesToDelete)

print(x_train[0])

# x_train = np.delete(x_train, trainIndexesToDelete)
# x_test = np.delete(x_test, testIndexesToDelete)
# y_train = np.delete(y_train, trainIndexesToDelete)
# y_test = np.delete(y_test, testIndexesToDelete)

# print(str(len(x_train)))


#x_train = x_train.reshape(len(y_train), 32, 32, 3)
#x_test = x_test.reshape(len(y_test), 32, 32, 3)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

print("Length of four-class training dataset: " + str(len(y_train)))
print("Length of four-class test dataset: " + str(len(y_test)))
