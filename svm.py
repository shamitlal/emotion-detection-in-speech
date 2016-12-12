
import numpy

def convertManyToOne(Y):
    newY = numpy.empty((0, 1))
    for i in xrange(len(Y)):
        for j in xrange(len(Y[i])):
            if Y[i][j] == 1:
                newY = numpy.vstack([newY, j])
                break
    return newY

import scipy.io as sio
'''
X = sio.loadmat('X_scaled.mat')['X_scaled']
Y = sio.loadmat('Y.mat')['Y']
Y = convertManyToOne(Y)
Y = numpy.hstack(Y)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.svm import SVC
clf = SVC(verbose=True, cache_size = 700)
clf.fit(X_train, y_train)

print "Training Accuracy %.2f\n" % (clf.score(X_train, y_train))
print "Testing Accuracy %.2f\n" % (clf.score(X_test, y_test))
'''

X_train = sio.loadmat('X_train.mat')['X_train']
Y_train = sio.loadmat('Y_train.mat')['Y_train']

X_test = sio.loadmat('X_test.mat')['X_test']
Y_test = sio.loadmat('Y_test.mat')['Y_test']

Y_test=numpy.hstack(Y_test)
Y_train=numpy.hstack(Y_train)
from sklearn.svm import SVC

clf = SVC(verbose=True, cache_size = 700)



clf.fit(X_train, Y_train)
print "Training Accuracy %.2f\n" % (clf.score(X_train, Y_train))
print "Testing Accuracy %.2f\n" % (clf.score(X_test, Y_test))

'''
import csv as csv
predictions_file = open("/Users/shamitlal/Desktop/shamit/sem 6/minor/speech-emotion-recognition-master/csv_1.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["solved_status"])
output = clf.predict(X_test).astype(int) 
print output
for i in output:
    open_file_object.writerow([i])
predictions_file.close()
'''

matrix=numpy.zeros((7,7))
for i in range(X_test.shape[0]):
	opclass= clf.predict(X_test[i])
	matrix[Y_test[i]][opclass[0]]+=1

for i in range(7):
	for j in range(7):
		print int(matrix[i][j]),
	print ''

print '\n\n'

for i in range(7):
	sum=0
	for j in range(7):
		sum+=(int)(matrix[i][j])
	for j in range(7):
		print("%.2f"%(matrix[i][j]*100/sum)),
	print ''
