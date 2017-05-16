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
from ML_treatment import componentAnalysis, normalizeData

print('libaries loaded')
# Close all open windows
plt.close()

# Chose dataset
#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
#raw_data = urllib.urlopen(url)
#dataset1 = np.loadtxt(raw_data)

dataset = np.loadtxt("Datasets/housing.data") #"Users/groenera/Desktop/file.csv"
print('data set loaded')

# Preprocessing
shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))

#componentAnalysis(dataset, 'raw') #plot preprocessing

normalizeData(dataset)

dataset = np.delete(dataset, 3, axis=1) # 4th dimesion = 3rd element

#componentAnalysis(dataset, 'normalized_woDim4')

#dataset = np.concatenate((dataset[:,-2],dataset[:,-1])).reshape(shape[0],2)

shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))


# Define Parameters
tt_ratio = 0.5

# generate data set
#Xc  = np.expand_dims(dataset[:,5], axis=1)
Xc  = np.array(dataset[:,0:shape[1]-1])
Yc  = np.array(dataset[:,shape[1]-1])

X,x,Y,y  = train_test_split(Xc,Yc,test_size = tt_ratio, random_state = 0)

# Define hyperparameters
gammaVal = 0.1

# ## train rvr
rvr = RVR(gamma = gammaVal, kernel = 'rbf')
mem_us_rvr = memory_usage((rvr.fit,(X,Y)),interval=0.1)
minMem_rvr = min(mem_us_rvr)
maxMem_rvr = max(mem_us_rvr)
#
t1 = time.time()
rvr.fit(X,Y)
t2 = time.time()
# #
rvr_err   = mean_squared_error(rvr.predict(x),y)
rvr_s      = np.sum(rvr.active_)
print "RVR -- NMSR {0}, # SV {1}, time {2}, min Memroy {3}, max Memory {4}".format(rvr_err, rvr_s, t2 - t1,minMem_rvr, maxMem_rvr)
#


## train svr
svr = GridSearchCV(SVR(kernel = 'rbf', gamma = gammaVal), param_grid = {'C':[0.001,0.1,1,10,100]},cv = 10)
mem_us_svr = memory_usage((svr.fit,(X,Y)),interval=0.1)
minMem_svr = min(mem_us_svr)
maxMem_svr = max(mem_us_svr)
t1 = time.time()
svr.fit(X,Y)
t2 = time.time()
#print(mem_us_svr)
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


## plot test vs .1predicted data
predDim = 0#5-1 # dim-1
plt.figure(figsize = (16,10))
plt.plot(X[:,predDim],Y,"k+",markersize = 3, label = "train data")
plt.plot(x[:,predDim],y,"b+",markersize = 3, label = "test data")

plt.plot(xPred[:,predDim],y_RVR_pred,"b", markersize = 3, label = "RVR prediction")
plt.plot(xPred[:,predDim],y_RVR_pred + np.sqrt(var_RVR_pred),"b--", markersize = 3, label = "y_hat +- std")
plt.plot(xPred[:,predDim],y_RVR_pred - np.sqrt(var_RVR_pred),"b--", markersize = 3)
plt.plot(X[rvr.active_,predDim],Y[rvr.active_],"bo",markersize = 14,  label = "relevant vectors")
#
## plot one standard deviation bounds
plt.plot(xPred[:,predDim],y_SVR_pred,"r", markersize = 3, label = "SVR prediction")
##plt.plot(xPred[:,0],y_SVR_pred + np.sqrt(var_RVR_pred),"c", markersize = 3, label = "y_hat +- std")
##plt.plot(xPred[:,0],y_SVR_pred - np.sqrt(var_RVR_pred),"c", markersize = 3)
plt.plot(X[svr.best_estimator_.support_,predDim],Y[svr.best_estimator_.support_],"ro",markersize = 8,  label = "support vectors")
#
#plt.plot(xPred[:,0],y_real,"k", markersize = 3, label = "real function")
#
plt.legend(loc=6)
#plt.title("rvr")
#plt.xlim([xMin, xMax])
plt.savefig('Fig/regressionPlot.png')
##plt.show()


print('program succesfully terminated')