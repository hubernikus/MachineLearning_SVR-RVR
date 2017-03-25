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

# Libraries 
import matplotlib.pyplot as plt # Plots

import numpy as np 
import csv 

import random # Random number generator
import time # Time measurements

# Basic math function
from math import pi, sin

# Machine Learning Database
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split

from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC

## -------

plt.close('all')

inFileName = 'Datasets/human-development/gender_development.csv'
with open(inFileName, "rb") as csvfile:
    datareader = csv.reader(csvfile, delimiter = ',')
    rowNumb = -1
    for row in datareader:
        rowNumb += 1

countryCode = []
countryName = []
countryGDI = []
countryHDI_f = []

rowNames = []

with open(inFileName, "rb") as csvfile:
    datareader = csv.reader(csvfile, delimiter = ',')
    rowCount = 0
    for row in datareader:
        if(rowCount==0):
            rowCount += 1
            rowNames.append(row)
        else:
            if(row[2] != '..'):
                countryGDI.append(float(row[2]))
                countryHDI_f.append(float(row[3]))

                rowCount += 1
                countryName.append(row[1])
countryGDI = np.transpose(np.array([countryGDI]))
print(type(countryGDI))
print(countryGDI.shape)

countryHDI_f = np.transpose(np.array([countryHDI_f]))
print(type(countryHDI_f))
print(countryHDI_f.shape)

# Change value
x_val = countryGDI
y_val = countryHDI_f

# Define Regression Function
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)
lsrvr = RVR(kernel = 'rbf', gamma=0.1) ### CHANGE TO RVR

# Proceed regression using Support Vector Regression (SVR)
t1 = time.time()
y_rbf = svr_rbf.fit(x_val, y_val).predict(x_val)
t2 = time.time()
t_svr_rbf = t2-t1
print('Support Vector Regression with RBF kernel takes {} s'.format(t_svr_rbf))

t1 = time.time()
y_lin = svr_lin.fit(x_val, y_val).predict(x_val)
t2 = time.time()
t_svr_lin = t2-t1
print('Support Vector Regression with linear  kernel takes {} s'.format(t_svr_lin))

t1 = time.time()
#y_poly = svr_poly.fit(x_val, y_val).predict(x_val)
t2 = time.time()
t_svr_poly = t2-t1
print('Support Vector Regression with polynomial kernel takes {} s'.format(t_svr_poly))

# Proceed reression using Relevance Vector Regression (RVR)
t1 = time.time()
y_rvr = rvr.fit(x_val,y_val).predict(x_val)
t2 = time.time()
t_rvr = t2-t1
print('Relevance Vector Regression takes {} s'.format(t_rvr))

# Plot Data
plt.scatter(x_val,y_val, color='red',label='Datapoints')
plt.hold('on')

# Regression Plot
lw = 2
plt.plot(x_val, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(x_val, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(x_val, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.plot(x_val, y_rvr, color='magenta', lw=lw, label='RVR using RBF')


# Plot specification
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('Support Vector Regression')
plt.legend()
plt.show() # display plot in cmd


# Computation Time 

# Computation Cost
   # HOW# HOW

# Precision


# Memory Cost
   # HOW

print('Succesfully finished the demo script!')
