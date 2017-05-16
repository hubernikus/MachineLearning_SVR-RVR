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

print('Start program')

# Machine learnging
from sklearn.utils.estimator_checks import check_estimator
from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection.GridSearchCV import GridSearchCV
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

print('Libaries loaded')
# Close all open windows
plt.close()


# Import Dataset (CSV)
dataset=pd.read_csv('Datasets/energydata_complete.csv', sep=',',header=1)
dataset = np.array(dataset)
dataset = dataset[:,1:-2]

# Crop for easy calcualtion
analysisDim = 0
componentAnalysisLast(dataset[1:10000,:], 'energyData_raw', analysisDim) #plot preprocessing
normalizeData(dataset)

dataset = dataset[0:1000,:] # crop for ease of calculation

print('Data set loaded with shape: {}'.format(dataset.shape))
plt.show()


# Define Parameters
tt_ratios = np.linspace(0.1,0.9, num=5)
gammaVals = np.logspace(0.001,100,num=2)
 

# Inititialize lists
rvr_errs = []
svr_errs = []
boxLabels = []

N_crossValid = 10
for itVal in range(tt_ratios.shape[0]):
    rvr_errs.append([])
    svr_errs.append([])
    boxLabels.append(str(int(tt_ratios[itVal]*100)) + '%')
    for Ncv in range(N_crossValid):
        tt_ratio = 0.5
        
        # generate data set
        Xc  = np.array(dataset[:,0:dataset.shape[1]-1])
        Yc  = np.array(dataset[:,dataset.shape[1]-1])
        
        X,x,Y,y  = train_test_split(Xc,Yc,test_size = tt_ratio, random_state = int(time.time()*10000)%4294967295)
        print('Number of training samples {}'.format(X.shape[0]))
        
        # Define hyperparameters
        gammaVal = 0.1
        C_vals = np.logspace(0.001, 100, 5)
        
        
        ### RVR
        # train rvr
        rvr = RVR(gamma = gammaVal, kernel = 'rbf')
        
        t1 = time.time()
        mem_us_rvr = memory_usage((rvr.fit,(X,Y)),interval=0.1)
        #rvr.fit(X,Y)
        t2 = time.time()
        minMem_rvr = min(mem_us_rvr)
        maxMem_rvr = max(mem_us_rvr)
        
        # test rvr
        rvr_errs[itVal].append(mean_squared_error(rvr.predict(x),y))
        rvr_s      = np.sum(rvr.active_)
        print "RVR -- NMSR {0}, # SV {1}, time {2}, min Memroy {3}, max Memory {4}".format(rvr_errs[itVal][Ncv], rvr_s, t2 - t1,minMem_rvr, maxMem_rvr)
        
        
        ### SVR
        # train svr
        svr = GridSearchCV(SVR(kernel = 'rbf', gamma = gammaVal), param_grid = {'C':C_vals},cv = 10)
        t1 = time.time()
        mem_us_svr = memory_usage((svr.fit,(X,Y)),interval=0.1) 
        #svr.fit(X,Y)
        t2 = time.time()
        
        minMem_svr = min(mem_us_svr)
        maxMem_svr = max(mem_us_svr)
        
        # test rvr
        svr_errs[itVal].append(mean_squared_error(svr.predict(x),y))
        svs     = svr.best_estimator_.support_vectors_.shape[0]
        print "SVR -- NMSR {0}, # SV {1}, time {2}, min Memory {3}, max. Memory {4}".format(svr_errs[itVal][Ncv], svs, t2 - t1, minMem_svr, maxMem_svr)
    

plt.figure(figsize = (8,5))
plt.boxplot(svr_errs, 0, 'gD', labels = boxLabels)
name = 'tt_ratios'
plt.savefig('Fig/crossValid_SVR_' + name + '.png')


plt.figure(figsize = (8,5))
plt.boxplot(rvr_errs, 0, 'gD', labels = boxLabels)
name = 'tt_ratios'
plt.savefig('Fig/crossValid_RVR_' + name + '.png')


print('Program terminated')