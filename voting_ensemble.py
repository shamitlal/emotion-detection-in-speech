import numpy as np
import pandas as pd
import pylab as p
import csv as csv
import scipy.io as sio

'''
1:svm
2:adaboost
3:randomforest
4:logistic
5:gradientboosting
'''

data_test=np.zeros((6,10036,1))
for i in range(1,6):
    ensemble = pd.read_csv("/Users/shamitlal/Desktop/shamit/sem 6/minor/speech-emotion-recognition-master/csv_"+(str)(i)+".csv")
    #print ensemble.shape
    data_test[i]=ensemble.values
    #print (data_test[i])

Y_test = sio.loadmat('Y_test.mat')['Y_test']
#print int(Y_test[2][0])
acc=0
for i in range(372):
    temp=np.zeros(10)
    maximum = 0
    for j in range(1,6):
        val=int(data_test[j][i][0])
        temp[val]+=1
        if j==3:
            temp[val]+=2

    maximum=0
    maxindex=0
    for j in range(8):
        if temp[j]>maximum:
            maximum=temp[j]
            maxindex=j

    if maxindex == int(Y_test[i][0]):
        acc+=1

print acc*100/372



#print data_train_x_numpy[0][0]








'''
run these on terminal

export LANG="it_IT.UTF-8"  
export LC_COLLATE="it_IT.UTF-8"  
export LC_CTYPE="it_IT.UTF-8"  
export LC_MESSAGES="it_IT.UTF-8"  
export LC_MONETARY="it_IT.UTF-8"  
export LC_NUMERIC="it_IT.UTF-8"  
export LC_TIME="it_IT.UTF-8"  
export LC_ALL=
'''