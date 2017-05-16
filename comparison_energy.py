#------------------------------------------------------------------------------
#
#   Advanced Machine Learning - SVR vs. RVR
#
#   Author: Lukas Huber
#
#
#   Supervisor: Billard Aude
#
#------------------------------------------------------------------------------

print('start program')

# Machine learnging
from sklearn.utils.estimator_checks import check_estimator
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import normalize

from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC

# Figure library
import matplotlib.pyplot as plt

# Math library
import numpy as np
from scipy.stats import norm # Calculate norm 

from memory_profiler import memory_usage # Memory usage of function

import time

import urllib # Read from link

import csv 
import pandas as pd # Import datafiles

# Personal libraries
from ML_treatment import componentAnalysis, normalizeData, componentAnalysisLast

print('libaries loaded')
# Close all open windows
plt.close()

# Chose dataset
#dataset = np.loadtxt("Datasets/housing.data") #"Users/groenera/Desktop/file.csv"
#dataset = np.genfromtxt("Datasets/energydata_complete.csv", dtype=None, skip_header=1) #"Users/groenera/Desktop/file.csv"

# Import Dataset (CSV)
dataset=pd.read_csv('Datasets/energydata_complete.csv', sep=',',header=1)
datasetRaw = np.array(dataset)


datasetRaw = datasetRaw[1:100,:]
analysisDim = 0
times = []
days = []
months = [31,28,31,30,31,30,31,31,30,31,30,31] # length of the months

for i in range(datasetRaw.shape[0]):
    print(i)
    date, timeDat = datasetRaw[i,0].split(' ')
    year, month, day = date.split('-')
    hour, minut, sec = timeDat.split(':')
    
    days.append(sum(months[0:int(month)])+int(day))
    times.append(int(hour)*3600+int(minut)*60+int(sec))

datTim = np.array([days, times]).reshape(datasetRaw.shape[0],2)
dataset = np.concatenate((datasetRaw[:,1:],datTim),axis=1)

print('Dataset shape: {}'.format(dataset.shape))

# Crop for easy calcualtion
anlysisDim = 1
componentAnalysisLast(dataset[1:1000,:], 'energyData_raw', anlysisDim) #plot preprocessing
analysisDim = 25 
componentAnalysisLast(dataset[1:1000,:], 'energyData_raw', anlysisDim) #plot preprocessing

dataset = dataset[0:200,0:]


print('data set loaded')

# Preprocessing
shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))


componentAnalysisLast(dataset, 'energyData_raw', analysisDim) #plot preprocessing

normalizeData(dataset)

#dataset = np.delete(dataset, 3, axis=1) # 4th dimesion = 3rd element

#componentAnalysis(dataset, 'energyData_normalized_woDim4')

#dataset = np.concatenate((dataset[:,-2],dataset[:,-1])).reshape(shape[0],2)

shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))


# Define Parameters
tt_ratio = 0.5

# generate data set
#Xc  = np.expand_dims(dataset[:,5], axis=1)
Xc  = np.array(dataset[:,0:shape[1]-1])
Yc  = np.array(dataset[:,shape[1]-1])

MAXINT = 4294967295 
X,x,Y,y  = train_test_split(Xc,Yc,test_size = tt_ratio, 
                            random_state =  int(time.time()*10000)%MAXINT)
print('Number of training samples {}'.format(X.shape[0]))

# Define hyperparameters
gammaVal = 0.1


# ## train rvr
rvr = RVR(gamma = gammaVal, kernel = 'rbf')

t1 = time.time()
mem_us_rvr = memory_usage((rvr.fit,(X,Y)),interval=0.1)
#rvr.fit(X,Y)
t2 = time.time()

minMem_rvr = min(mem_us_rvr)
maxMem_rvr = max(mem_us_rvr)

rvr_err   = mean_squared_error(rvr.predict(x),y)
rvr_s      = np.sum(rvr.active_)
print "RVR -- NMSR {0}, # SV {1}, time {2}, min Memroy {3}, max Memory {4}".format(rvr_err, rvr_s, t2 - t1,minMem_rvr, maxMem_rvr)
#


## train svr
svr = GridSearchCV(SVR(kernel = 'rbf', gamma = gammaVal), param_grid = {'C':[0.001,0.1,1,10,100]},cv = 10)
t1 = time.time()
mem_us_svr = memory_usage((svr.fit,(X,Y)),interval=0.1) 
#svr.fit(X,Y)
t2 = time.time()

minMem_svr = min(mem_us_svr)
maxMem_svr = max(mem_us_svr)

svm_err = mean_squared_error(svr.predict(x),y)
svs     = svr.best_estimator_.support_vectors_.shape[0]
print "SVR -- NMSR {0}, # SV {1}, time {2}, min Memory {3}, max. Memory {4}".format(svm_err, svs, t2 - t1, minMem_svr, maxMem_svr)


# Time
# Memory usage
# Number of SV

# Create Prediction
dimTrain = Xc.shape
nPred = 1000

xPred = np.zeros((nPred,dimTrain[1]))
for i in range(dimTrain[1]):
    xPred[:,i] = np.linspace(min(Xc[:,i]),max(Xc[:,i]),nPred).reshape((nPred,))

y_RVR_pred, var_RVR_pred = rvr.predict_dist(xPred)

y_SVR_pred = svr.predict(xPred)



print('program succesfully terminated')