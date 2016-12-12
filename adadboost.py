
import numpy
import scipy.io as sio
from sklearn.ensemble import AdaBoostClassifier
X_train = sio.loadmat('X_train.mat')['X_train']
Y_train = sio.loadmat('Y_train.mat')['Y_train']

X_test = sio.loadmat('X_test.mat')['X_test']
Y_test = sio.loadmat('Y_test.mat')['Y_test']

Y_test=numpy.hstack(Y_test)
Y_train=numpy.hstack(Y_train)

clf = AdaBoostClassifier()
clf = clf.fit(X_train , Y_train)
print "\n\n predicting ... \n\n"

print "Training Accuracy %.2f\n" % (clf.score(X_train, Y_train))
print "Testing Accuracy %.2f\n" % (clf.score(X_test, Y_test))

import csv as csv
predictions_file = open("/Users/shamitlal/Desktop/shamit/sem 6/minor/speech-emotion-recognition-master/csv_2.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["solved_status"])
output = clf.predict(X_test).astype(int) 
print output
for i in output:
    open_file_object.writerow([i])
predictions_file.close()
