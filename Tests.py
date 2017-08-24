# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:02:34 2017

With the agreement of Mickael Rabouille, author of the original Matlab code

@author: translation to python 3.5 Sarah Juricic
"""
import numpy as np
#rng shuffle #initialisation du générateur random : utile en python?
import matplotlib.pyplot as plt
from math import pi as pi
from random import randint as randint


#==============================================================================
#                      Test function : ISHIGAMI
#==============================================================================
def ishigami_function():
    """
    ISHIGAMI function
    Crestaux et al. (2007) and Marrel et al. (2009) use: a = 7 and b = 0.1.
    Sobol' & Levitan (1999) use: a = 7 and b = 0.05.
    """
    a = 7
    b = 0.05
    f = lambda X:np.sin(X[:,0]) + a*np.sin(X[:,1])**2 + b*X[:,2]**4*np.sin(X[:,0])
    
    E = a/2 #??? à quoi sert E ???
    Vx1 = 1/2*(1 + b*(pi**4)/5)**2
    Vx2 = a**2/8
    Vx13 = 8*b**2*pi**8/225
    V = Vx1 + Vx2 + Vx13
    exact = np.array([[Vx1/V, 0, 0],
                       [0, Vx2/V, 0],
                       [Vx13/V, 0, 0]])
    exactDiag = exact.diagonal()
    return f, exactDiag

#==============================================================================
#                       testing algorithm configurations
#==============================================================================

#==================== Effect of bias ======================================
def bias_effect():
    """
    Compare the biased and unbiased sentivity analysis results
        defaults with the Ishigami function
        to do : plot comparison for any case studies
    returns a plot of the evolution of the results depending on the simulation number
    """
    from math import pi as pi
    from random import randint as randint
    warning_on = False
    ninput = 3 # def: X=[x1,x2,x3] -> xi=U(-pi,pi)
    SIc = np.zeros((ninput,450))
    SI = np.zeros((ninput,450))
    f,exactDiag = ishigami_function()
    
    #from 50 to 500 simulations
    for N in range(50,500):
        #create random model inputs of N simulations
        X = -pi + 2*pi*np.random.rand(N,ninput)
        #calculate the corresponding Ishigami function output
        Y = f(X).reshape((f(X).shape[0],f(X)[0].size))
        #calculate the corresponding sensitivity analysis...
        tSI,tSIc = rbdfast(Y, x = X, warning_on = warning_on)
        #... and reshape the results into plottable array
        SI[:,N-50],SIc[:,N-50] = tSI.reshape((1,3)),tSIc.reshape((1,3))
    
    # Print plot : effect of bias
    plt.plot(SI.transpose(),'r--')
    plt.plot(SIc.transpose(),'b--')
    plt.plot([[exactDiag.item(i) for i in range(0,3)] for k in range(0,450)],
               color = '#003366',
               linewidth = 2.0)
    plt.title('Effect of bias')
    plt.ylabel('SI')
    plt.xlabel('Simulation Number')
    plt.show()
    return


#======================== Effect of M value ===============================
def effect_threshold():
    """
    First order indices are calculated from the first m frequencies of the FFT
    Plot with the ishigami function results by default
        To do : make any sensitivity analysis result plottable
    returns plot of the impact of changing this threshold value
    """
    f,exactDiag = ishigami_function()
    ninput = 3
    SIc = np.zeros((ninput,30))
    SI = np.zeros((ninput,30))
    x = -pi + 2*pi*np.random.rand(500,ninput)
    y = f(x).reshape((f(x).shape[0],f(x)[0].size))
    for m in range(1,31):
        tempSI,tempSIc = rbdfast(y, x = x, m = m)
        SI[:,m-1],SIc[:,m-1] = tempSI.reshape((1,3)),tempSIc.reshape((1,3))
    
    plt.plot(SIc.transpose(),'b')
    plt.plot(SI.transpose(),'r')
    plt.plot([[exactDiag[i] for i in range(0,3)] for k in range(0,30)],'k')
    plt.title('Effect of the M value')
    plt.ylabel('SI')
    plt.xlabel('M value')
    plt.show()
    return


