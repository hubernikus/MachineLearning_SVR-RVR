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
from ML_treatment import componentAnalysis, normalizeData, removeDimension

# Close all open windows
plt.close()

# Chose dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
raw_data = urllib.urlopen(url)
dataset = np.loadtxt(raw_data)

# Preprocessing
shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))

#componentAnalysis(dataset, 'raw') #plot preprocessing

normalizeData(dataset)

for i in range(5):
    dataset = np.delete(dataset, 0, axis=1) # 3rd dimesion = 2nd element

#dataset = np.delete(dataset, 2, axis=1) # 3rd dimesion = 2nd element

componentAnalysis(dataset, 'normalized_woDim')

shape = dataset.shape
print("Dataset shape is {}x{}".format(shape[0],shape[1]))


# Define Parameters
tt_ratio = 0.5

# generate data set
#Xc  = np.expand_dims(dataset[:,5], axis=1)
Xc  = np.array(dataset[:,0:shape[1]-1])
Yc  = np.array(dataset[:,shape[1]-1])

X,x,Y,y  = train_test_split(Xc,Yc,test_size = tt_ratio, random_state = 0)

print(X.shape); print(Y.shape)

## train svr
rvr = SVR(gamma = 1,kernel = 'rbf')
mem_us_rvr = memory_usage((rvr.fit,(X,Y)),interval=0.1)
#print(mem_us_rvr)
t1 = time.time()

# ## train rvrpuudii dich hani vor de 9ee glaubs am bhf / zkb gseh aber bisch zwiit weg gsi zum rüefee
rvr = RVR(gamma = 1, kernel = 'rbf')
mem_us_rvr = memory_usage((rvr.fit,(X,Y)),interval=0.1)
print(mem_us_rvr)
#
t1 = time.time()
rvr.fit(X,Y)
t2 = time.time()
# #
rvr_err   = mean_squared_error(rvr.predict(x),y)
rvr_s      = np.sum(rvr.active_)
print "RVR error on test set is {0}, number of relevant vectors is {1}, time {2}".format(rvr_err, rvr_s, t2 - t1)
#


## train svr
svr = GridSearchCV(SVR(kernel = 'rbf', gamma = 1), param_grid = {'C':[0.001,0.1,1,10,100]},cv = 10)
mem_us_svr = memory_usage((svr.fit,(X,Y)),interval=0.1)
t1 = time.time()
svr.fit(X,Y)
t2 = time.time()
#print(mem_us_svr)
svm_err = mean_squared_error(svr.predict(x),y)
svs     = svr.best_estimator_.support_vectors_.shape[0]
print "SVM error on test set is {0}, number of support vectors is {1}, time {2}".format(svm_err, svs, t2 - t1)


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
plt.plot(xPred[:,0],y_SVR_pred,"r", markersize = 3, label = "SVR prediction")
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
#

#n_grid = 100
#max_x      = np.max(X,axis = 0)
#min_x      = np.min(X,axis = 0)
#max_y      = np.max(Y)
#min_y      = np.min(Y)
#X1         = np.linspace(min_x,max_x,n_grid)
#Y1         = np.linspace(min_y,max_y,n_grid)
#x1,y1      = np.meshgrid(X1,Y1)
#Xgrid      = np.zeros([n_grid**2,2])
#Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
#Xgrid[:,1] = np.reshape(y1,(n_grid**2,))
#XgridExp = np.expand_dims(Xgrid[:,0],axis=1)
#
#mu,var     = rvr.predict_dist(XgridExp)
#
##mu,var     = rvr.predict_dist(np.expand_dims(Xgrid[:,0],axis =1))
#
#probs      = norm.pdf(Xgrid[:,1],loc = mu, scale = np.sqrt(var))
#plt.figure(figsize = (12,8))
#plt.contourf(X1,Y1,np.reshape(probs,(n_grid,n_grid)),cmap="coolwarm")
##plt.plot(X1,10*np.sinc(X1),'k-',linewidth = 3, label = 'real function')
##plt.plot(X1,10*np.sinc(X1)-1.96,'k-',linewidth = 2, label = '95% real lower bound',
##         linestyle = '--')
##plt.plot(X1,10*np.sinc(X1)+1.96,'k-',linewidth = 2, label = '95% real upper bound',
##         linestyle = '--')
#plt.plot(rvr.relevant_vectors_,Y[rvr.active_],"co",markersize = 12,  label = "relevant vectors")
#plt.title("PDF of Predictive Distribution of Relevance Vector Regression")
#plt.colorbar()
#plt.legend()
#plt.show()

print('Succesfully finished the demo script!')