#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:50:13 2017

@author: luc
"""


import matplotlib.pyplot as plt

import numpy as np
    
def componentAnalysis(dataSet, name='0', badDimensions=[]):
    shape = dataSet.shape

    
    plt.figure(figsize = (16,10))
    for i in range(shape[1]):#shape[0]*shope[1]):
#        while(i in badDimensions):
#            i = i+1
#        if(i>=shape[1]): break
        
        
        for j in range(shape[1]):
#            while(j in badDimensions):
#                j = j+1
#            if(j >= shape[1]): break# 
            
            index = i*shape[1]+(j+1)
            plt.subplot(shape[1],shape[1],index)
            plt.plot(dataSet[:,i],dataSet[:,j],"b+")
        #plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    
    plt.savefig('Fig/componentPlot_' + name + '.png')




def componentAnalysisLast(dataSet, nameSpec='', regrDimension=-1,  badDimensions=[]):
    shape = dataSet.shape

    nCol = int(np.ceil(np.sqrt(shape[1])))
    nRow = int(np.ceil(1.*(shape[1])/nCol))
    
    plt.figure(figsize = (16,10))
    
    for i in range(shape[1]):#shape[0]*shope[1]):
        #index = shape[1]%nCol+np.floor(shape[1]/nCol)*nCol
            
        plt.subplot(nRow,nCol,i+1)
        plt.plot(dataSet[:,i],dataSet[:,regrDimension],"b+")

    plt.savefig('Fig/componentPlot_' + nameSpec + '.png')
    print('Horizontal analysis figure finished. ')




def normalizeData(dataSet):
    shape = dataSet.shape

    for i in range(shape[1]):
        mean = np.mean(dataSet[:,i])
        std = np.std(dataSet[:,i])

        if(std): # nonzero check
            dataSet[:,i] = (dataSet[:,i]-mean)/std
        else:
            dataSet[:,i] = (dataSet[:,i]-mean)
        





def removeDimension(dataSet, dims = []):
    
    for r in dims:
        dataSet = np.delete(dataSet, (r-1), axis=1)
        # print('Row {} removed'.format(r))
   
        

