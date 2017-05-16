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

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm # Calculate norm 

from memory_profiler import memory_usage # Memory usage of function

import time

import urllib

# Personal libraries
from ML_treatment import componentAnalysis, normalizeData, componentAnalysisLast

print('libaries loaded')
# Close all open windows
plt.close()

#dataset = np.loadtxt("Datasets/housing.data") #"Users/groenera/Desktop/file.csv"
print('data set loaded')

# Preprocessing
shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))

normalizeData(dataset)
componentAnalysisLast(dataset)
dataset = np.delete(dataset, 3, axis=1) # 4th dimesion = 3rd element
dataset = dataset[:,:] # reduce dataset, remove for large calculations

shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))


# Wine dataset
dataset = np.loadtxt("Datasets/winequality-white.csv",delimiter=';',skiprows=1) #"Users/groenera/Desktop/file.csv"
normalizeData(dataset)
print('Data set loaded with shape: {}'.format(dataset.shape))

# Define Parameters
tt_ratios = np.linspace(0.1,0.9, num=2)
gammaVals = np.logspace(0.001,100,num=2)
dimMax = dataset.shape[1]
nDim = range(2,dataset.shape[1]-1)

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

N_crossValid = 3

for itVal in range(0,dataset.shape[1]-3):
    rvr_errs.append([])
    svr_errs.append([])
    
    t_train_svr.append([])
    t_train_rvr.append([])

    t_test_svr.append([])
    t_test_rvr.append([])

    n_svr.append([])
    n_rvr.append([])
    
    boxLabels.append('n = ' + str(int(nDim[itVal])) )
    for Ncv in range(N_crossValid):
        tt_ratio = 0.5
        
        # generate data set
        Xc  = np.array(dataset[:,2:itVal+3])
        Yc  = np.array(dataset[:,dataset.shape[1]-1])
        
        X,x,Y,y  = train_test_split(Xc,Yc,test_size = tt_ratio, random_state = int(time.time()*10000)%4294967295)
        print('Number of training samples {}'.format(X.shape[0]))
        
        # Define hyperparameters
        gammaVal = 0.1
        C_vals = np.logspace(0.01, 100, 5)
        
        
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
        
        svr = GridSearchCV(SVR(kernel = 'rbf', gamma = gammaVal), param_grid = {'C':[0.001,0.1,1,10,100]},cv = 10)
        

        t1 = time.time()
        mem_us_svr = memory_usage((svr.fit,(X,Y)),interval=0.1) 
#        svr.fit(X,Y)
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
dataName = 'housing_NMRS'
meanSVR = np.mean(np.array(svr_errs),axis=1)
stdSVR = np.std(np.array(svr_errs))
meanRVR = np.mean(np.array(rvr_errs),axis=1)
stdRVR = np.std(np.array(rvr_errs))

plt.figure(figsize = (8,5))
plt.plot(nDim,meanSVR, c='b', label='SVR')
plt.errorbar(nDim, meanSVR, stdSVR, c='b')
plt.plot(nDim,meanRVR, c='r', label='RVR')
plt.errorbar(nDim, meanRVR, stdRVR, c='r')
plt.xlabel('Number of Dimension', fontsize=18)
plt.ylabel('NMSE', fontsize=18)
plt.legend()
plt.savefig('Fig/crossValid_SVR_' + dataName + '.png')


#
dataName = 'housing_time'

meanRVR = np.mean(np.array(t_train_rvr),axis=1)
stdRVR = np.std(np.array(t_train_rvr))
meanSVR = np.mean(np.array(t_train_svr),axis=1)
stdSVR = np.std(np.array(t_train_svr))

plt.figure(figsize = (8,5))
plt.plot(nDim,meanSVR, c='b', label='SVR')
plt.errorbar(nDim, meanSVR, stdSVR, c='b')
plt.plot(nDim,meanRVR, c='r', label='RVR')
plt.errorbar(nDim, meanRVR, stdRVR, c='r')
plt.xlabel('Number of Dimension', fontsize=18)
plt.ylabel('Time [s]', fontsize=18)
plt.legend()
plt.savefig('Fig/crossValid_SVR_' + dataName + '.png')


#
dataName = 'housing_SV'
meanRVR = np.mean(np.array(n_rvr),axis=1)
stdRVR = np.std(np.array(n_rvr))
meanSVR = np.mean(np.array(n_svr),axis=1)
stdSVR = np.std(np.array(n_svr))

plt.figure(figsize = (8,5))
plt.plot(nDim,meanSVR, c='b', label='SVR')
plt.errorbar(nDim, meanSVR, stdSVR, c='b')
plt.plot(nDim,meanRVR, c='r', label='RVR')
plt.errorbar(nDim, meanRVR, stdRVR, c='r')
plt.xlabel('Number of Dimension', fontsize=18)
plt.ylabel('Number of support vectors', fontsize=18)
plt.legend()
plt.savefig('Fig/crossValid_SVR_' + dataName + '.png')

#
plt.figure(figsize = (8,5))
plt.plot(nDim,meanRVR, c='r', label='RVR')
plt.errorbar(nDim, meanRVR, stdRVR, c='r')
plt.xlabel('Number of Dimension', fontsize=18)
plt.ylabel('Number of support vectors', fontsize=18)
plt.legend()
plt.savefig('Fig/crossValid_SVR_' + dataName + '.png')


    
#plt.plot(np.array(rvr_errs).T)
#plt.grid('on')

print('program succesfully terminated')