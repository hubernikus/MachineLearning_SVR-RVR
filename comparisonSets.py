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

from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm # Calculate norm 

from memory_profiler import memory_usage # Memory usage of function

import time

#import sys # Variable size

#%matplotlib inline

plt.close()
# parameters
n = 100

# generate data set
xMin, xMax = -5, 5
np.random.seed(5)
Xc       = np.ones([n,1])
Xc[:,0]  = np.linspace(xMin,xMax,n)
Yc       = 10*np.sinc(Xc[:,0]) + np.random.normal(0,1,n)
X,x,Y,y  = train_test_split(Xc,Yc,test_size = 0.5, random_state = 0)

# train rvr
rvr = RVR(gamma = 1,kernel = 'rbf')
t1 = time.time()
rvr.fit(X,Y)
#mem_usage_rvr = memory_usage(rvr.fit(X,Y))
t2 = time.time()

rvr_err   = mean_squared_error(rvr.predict(x),y)
rvs       = np.sum(rvr.active_)
print "RVR error on test set is {0}, number of relevant vectors is {1}, time {2}".format(rvr_err, rvs, t2 - t1)

# train svr
svr = GridSearchCV(SVR(kernel = 'rbf', gamma = 1), param_grid = {'C':[0.001,0.1,1,10,100]},cv = 10)
t1 = time.time()
svr.fit(X,Y)
mem_usage_svr = memory_usage(svr.fit(X,Y))
t2 = time.time()
svm_err = mean_squared_error(svr.predict(x),y)
svs     = svr.best_estimator_.support_vectors_.shape[0]
print "SVM error on test set is {0}, number of support vectors is {1}, time {2}".format(svm_err, svs, t2 - t1)


# Create Prediction
nPred = 1000
xPred = np.linspace(min(Xc),max(Xc),nPred).reshape((nPred,1))
y_RVR_pred, var_RVR_pred = rvr.predict_dist(xPred)
#y_SVR_pred, var_SVR_pred = svr.predict_dist(xPred) NOT possible?
y_SVR_pred= svr.predict(xPred)
y_real = 10*np.sinc(xPred)


# plot test vs predicted datax
plt.figure(figsize = (16,10))
plt.plot(X[:,0],Y,"k+",markersize = 3, label = "train data")
plt.plot(x[:,0],y,"b+",markersize = 3, label = "test data")

plt.plot(xPred[:,0],y_RVR_pred,"b", markersize = 3, label = "RVR prediction")
#plt.plot(xPred[:,0],y_RVR_pred + np.sqrt(var_RVR_pred),"c", markersize = 3, label = "y_hat +- std")
#plt.plot(xPred[:,0],y_RVR_pred - np.sqrt(var_RVR_pred),"c", markersize = 3)
plt.plot(rvr.relevant_vectors_,Y[rvr.active_],"co",markersize = 14,  label = "relevant vectors")

# plot one standard deviation bounds
plt.plot(xPred[:,0],y_SVR_pred,"r", markersize = 3, label = "SVR prediction")
#plt.plot(xPred[:,0],y_SVR_pred + np.sqrt(var_RVR_pred),"c", markersize = 3, label = "y_hat +- std")
#plt.plot(xPred[:,0],y_SVR_pred - np.sqrt(var_RVR_pred),"c", markersize = 3)
plt.plot(svr.best_estimator_.support_vectors_,Y[svr.best_estimator_.support_],"ro",markersize = 8,  label = "support vectors")

plt.plot(xPred[:,0],y_real,"k", markersize = 3, label = "real function")

plt.legend()
plt.title("rvr")
plt.xlim([xMin, xMax])
#plt.show()


n_grid = 100
max_x      = np.max(X,axis = 0)
min_x      = np.min(X,axis = 0)
max_y      = np.max(Y)
min_y      = np.min(Y)
X1         = np.linspace(min_x,max_x,n_grid)
Y1         = np.linspace(min_y,max_y,n_grid)
x1,y1      = np.meshgrid(X1,Y1)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(y1,(n_grid**2,))
XgridExp = np.expand_dims(Xgrid[:,0],axis=1)

mu,var     = rvr.predict_dist(XgridExp)

#mu,var     = rvr.predict_dist(np.expand_dims(Xgrid[:,0],axis =1))

probs      = norm.pdf(Xgrid[:,1],loc = mu, scale = np.sqrt(var))
plt.figure(figsize = (12,8))
plt.contourf(X1,Y1,np.reshape(probs,(n_grid,n_grid)),cmap="coolwarm")
#plt.plot(X1,10*np.sinc(X1),'k-',linewidth = 3, label = 'real function')
#plt.plot(X1,10*np.sinc(X1)-1.96,'k-',linewidth = 2, label = '95% real lower bound',
#         linestyle = '--')
#plt.plot(X1,10*np.sinc(X1)+1.96,'k-',linewidth = 2, label = '95% real upper bound',
#         linestyle = '--')
plt.plot(rvr.relevant_vectors_,Y[rvr.active_],"co",markersize = 12,  label = "relevant vectors")
plt.title("PDF of Predictive Distribution of Relevance Vector Regression")
plt.colorbar()
plt.legend()
#plt.show()

print('Succesfully finished the demo script!')
