import numpy
from sklearn.cross_validation import train_test_split
import essentia
import essentia.standard
import os
import scipy.io.wavfile
import scipy.io
from essentia.standard import *
import numpy
from sklearn import preprocessing

def convertManyToOne(Y):
    newY = numpy.empty((0, 1))
    for i in xrange(len(Y)):
        for j in xrange(len(Y[i])):
            if Y[i][j] == 1:
                newY = numpy.vstack([newY, j])
                break
    return newY

import scipy.io as sio
X = sio.loadmat('X_scaled.mat')['X_scaled']
Y = sio.loadmat('Y.mat')['Y']
Y = convertManyToOne(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

scipy.io.savemat('X_train.mat', {'X_train' : X_train})
scipy.io.savemat('Y_train.mat', {'Y_train' : Y_train})

scipy.io.savemat('X_test.mat', {'X_test' : X_test})
scipy.io.savemat('Y_test.mat', {'Y_test' : Y_test})
