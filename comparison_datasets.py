#------------------------------------------------------------------------------
#
#   Advanced Machine Learning - SVR vs. RVR
#
#   Author: Lukas Huber
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



# Define Parameters
tt_ratios = np.linspace(0.1,0.9, num=2)
gammaVals = np.logspace(0.001,100,num=5)

tt_ratios = [0.5]
itVal = 0

# Inititialize lists
rvr_errs = []
svr_errs = []

t_train_svr = []
t_train_rvr = []

t_test_svr = []
t_test_rvr = []

n_svr = []
n_rvr = []

boxLabels = []

N_crossValid = 10

dataNames = ['wine','housing','energy']
regrDim = [1,-1,1]
for itVal in range(len(dataNames)):
    dataName = dataNames[itVal]
    
    if(dataName=='wine'):
        # Wine dataset
        dataset = np.loadtxt("Datasets/winequality-white.csv",delimiter=';',skiprows=1) #"Users/groenera/Desktop/file.csv"
        normalizeData(dataset)
        print('Data set loaded with shape: {}'.format(dataset.shape))
    elif(dataName=='housing'):
        # Housing dataset
        dataset = np.loadtxt("Datasets/housing.data") #"Users/groenera/Desktop/file.csv"
        dataset = normalize(dataset)
    elif(dataName=='energy'):
        # Housing dataset
        dataset=pd.read_csv('Datasets/energydata_complete.csv', sep=',',header=1)
        dataset = np.array(dataset)
        times = []
        days = []
        months = [31,28,31,30,31,30,31,31,30,31,30,31] # length of the months
        
        for i in range(dataset.shape[0]): # Change date/time string to values
            date, timeDat = dataset[i,0].split(' ')
            year, month, day = date.split('-')
            hour, minut, sec = timeDat.split(':')
            
            days.append(sum(months[0:int(month)])+int(day))
            times.append(int(hour)*3600+int(minut)*60+int(sec))
        
        datTim = np.array([days, times]).reshape(dataset.shape[0],2)
        dataset = np.concatenate((dataset[:,1:],datTim),axis=1)
        normalizeData(dataset)
        print('Data set loaded with shape: {}'.format(dataset.shape))
        
    limit = 100 # zero if no restriction
    if(limit):
        dataset = dataset[0:limit,:]

    rvr_errs.append([])
    svr_errs.append([])
    
    t_train_svr.append([])
    t_train_rvr.append([])

    t_test_svr.append([])
    t_test_rvr.append([])

    n_svr.append([])
    n_rvr.append([])
    
    #boxLabels.append(str(int(tt_ratios[itVal]*100)) + '%')
    
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
        t_train_rvr[itVal].append(t2-t1)
        minMem_rvr = min(mem_us_rvr)
        maxMem_rvr = max(mem_us_rvr)
        
        # test rvr
        t1= time.time()
        rvr_errs[itVal].append(mean_squared_error(rvr.predict(x),y))
        t2 = time.time()
        t_test_rvr[itVal].append(t2-t1)
        
        rvr_s      = np.sum(rvr.active_)
        n_rvr[itVal].append(rvr_s)
        print "RVR -- NMSR {0}, # SV {1}, train time {2}, test time {3}, min Memroy {4}, max Memory {5}".format(rvr_errs[itVal][Ncv], rvr_s,t_train_rvr[itVal][Ncv], t2 - t1,minMem_rvr, maxMem_rvr)
        
        ### SVR
        # train svr
        svr = GridSearchCV(SVR(kernel = 'rbf', gamma = gammaVal), param_grid = {'C':C_vals},cv = 10)
        t1 = time.time()
        mem_us_svr = memory_usage((svr.fit,(X,Y)),interval=0.1) 
        #svr.fit(X,Y)
        t2 = time.time()
        t_train_svr[itVal].append(t2-t1)
        
        minMem_svr = min(mem_us_svr)
        maxMem_svr = max(mem_us_svr)
        
        # test svr
        t1 = time.time()
        svr_errs[itVal].append(mean_squared_error(svr.predict(x),y))
        t2 = time.time()
        t_test_svr[itVal].append(t2-t1)
        
        svs     = svr.best_estimator_.support_vectors_.shape[0]
        n_svr[itVal].append(svs)
        
        print "SVR -- NMSR {0}, # SV {1}, train time {2}, test time {3}, min Memory {4}, max. Memory {5}".format(svr_errs[itVal][Ncv], svs, t_train_svr[itVal][Ncv],t2-t1, minMem_svr, maxMem_svr)
 
    
#%%

plt.figure(figsize = (8,5))

plt.plot(np.array(svr_errs).T)
name = 'tt_ratios'
plt.title('NMSE - SVR')
plt.savefig('Fig/crossValid_SVR_' + dataName + '_' + name + '.png')

#out = csv.writer(open("Results/AnalysisDatasets.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
#out.writerow(data)

print('Program terminated')