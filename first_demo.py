#------------------------------------------------------------------------------
#   Author: Lukas Huber
#
#
#   Supervisor: Billard Aude
#
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

import numpy as np # Plots
import csv 

import random # Random number generator
import time # Time measurements

# Basic math function
from math import pi, sin

# Machine Learning Database
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split

#from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC

plt.close('all')


#fileIn = open('workfile', 'w')
#for line in fileIn:
#    x = 1


# Create presonal sample
N_samp = 100; # Sample size
x_min = 0
x_max = 2*pi
#x_val = [random.random() for _ in range(N_samp)]

# Create random random pseudo-measurement in form of sinus curve with noise          
x_val = np.sort(2*pi*np.random.rand(N_samp,1),axis=0);
#y_val = np.array(np.random.rand(N_samp,1)) + np.sin(x_val).ravel()
y_val = np.sin(x_val).ravel() + np.ravel(np.random.normal(0,0.5,(N_samp,1)))
                

# Create reference sinus curve with 100 steps         
N_range = 100         
x2_val = [i/N_range*2*pi for i in range(N_range)]


# Define Regression Function
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)


# Proceed regression using Support Vector Regression (SVR)
y_rbf = svr_rbf.fit(x_val, y_val).predict(x_val)
y_lin = svr_lin.fit(x_val, y_val).predict(x_val)
y_poly = svr_poly.fit(x_val, y_val).predict(x_val)

# Proceed reression using Relevance Vector Regression (RVR)
rvm = SVR(kernel = 'rbf', gamma=1) ### CHANGE TO RVR
t1 = time.time()
y_rvr = rvm.fit(x_val,y_val).predict(x_val)
t2 = time.time()

t_rvr = t2 -t1
print('Relevance Vector Regression takes {} s'.format(t_rvr))


# Plot Data
plt.scatter(x_val,y_val, color='red',label='Datapoints')
plt.hold('on')
plt.xlim([x_min,x_max])
plt.plot(x2_val, [sin(x2_val[i]) for i in range(len(x2_val))], c='k',label='Original function')         

# Regression Plot
lw = 2
plt.plot(x_val, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(x_val, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(x_val, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')

plt.plot(x_val, y_rvr, color='magenta', lw=lw, label='RVR using RBF')


# Plot specification
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()





# Computation Time 
 
# Computation Cost


# Precision




# Memory Cost

print('Finish Demo Script')
