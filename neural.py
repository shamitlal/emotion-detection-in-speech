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

X_scaled = sio.loadmat('X_scaled.mat')['X_scaled']
Y = sio.loadmat('Y.mat')['Y']
X = sio.loadmat('X_scaled.mat')['X_scaled'] ## ADDED X HERE
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter 
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.utilities           import percentError
num_inputs = len(X[0])
ds = ClassificationDataSet(num_inputs, 1 , nb_classes=7)

Y = convertManyToOne(Y)

for k in xrange(len(X)): 
	ds.addSample(X_scaled[k],Y[k])

ds._convertToOneOfMany()
tstdata, trndata = ds.splitWithProportion( 0.25 ) #25% test data


fnn = buildNetwork( trndata.indim, 400 ,200, trndata.outdim, outclass=SoftmaxLayer )

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 

NUM_EPOCHS = 200
for i in range(NUM_EPOCHS):
    error = trainer.train()
    print "Epoch: %d, Error: %7.4f" % (i, error)

    #train error calculation
    total = true = 0.0
    for x, y in trndata:
        out = fnn.activate(x).argmax()
        if out == y.argmax():
            true+=1
        #print str(out) + " " + str(y.argmax())
        total+=1
    res = true/total
    print "Accuracy on training data %.2f percent\n" % (res*100.0)

    #test error calculation
    total = true = 0.0
    for x, y in tstdata:
        out = fnn.activate(x).argmax()
        if out == y.argmax():
            true+=1
        #print str(out) + " " + str(y.argmax())
        total+=1
    res = true/total
    print "Accuracy on test data %.2f percent\n" % (res*100.0)
    

#error calculation
total = true = 0.0
for x, y in trndata:
    out = fnn.activate(x).argmax()
    if out == y.argmax():
        true+=1
    #print str(out) + " " + str(y.argmax())
    total+=1
res = true/total
print "Accuracy on training data %.2f percent\n" % (res*100.0)
	
#error calculation
total = true = 0.0
for x, y in tstdata:
    out = fnn.activate(x).argmax()
    if out == y.argmax():
        true+=1
    #print str(out) + " " + str(y.argmax())
    total+=1
res = true/total
print "Accuracy on test data %.2f percent\n" % (res*100.0)
