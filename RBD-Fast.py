# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:53:15 2017


"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#==============================================================================
#           Test function RBD FAST : ISHIGAMI
#==============================================================================
def rbdfast_test():
    """
    ISHIGAMI fonction
    Crestaux et al. (2007) and Marrel et al. (2009) use: a = 7 and b = 0.1.
    Sobol' & Levitan (1999) use: a = 7 and b = 0.05.
    """
    a = 7
    b = 0.05
    pi = np.math.pi
    f = lambda X:np.sin(X[:,0]) + a*np.sin(X[:,1])**2 + b*X[:,2]**4*np.sin(X[:,0])
    ninput = 3 # def: X=[x1,x2,x3] -> xi=U(-pi,pi)
    from math import pi as pi
    from random import randint as randint
    E = a/2 #??? à quoi sert E ???
    Vx1 = 1/2*(1 + b*pi**4/5)**2
    Vx2 = a**2/8
    Vx13 = b**2*pi**8/225
    V = Vx1 + Vx2 + Vx13
    exact = np.matrix([[Vx1/V, 0, 0],
                       [0, Vx2/V, 0],
                       [Vx13/V, 0, 0]])
    exactDiag = exact.diagonal()
    
    #rng shuffle #initialisation du générateur random : utile en python?
    
    #==================== Effect of bias ======================================
    SIc = np.zeros((ninput,450))
    SI = np.zeros((ninput,450))
    #warning('off','RBD:lowSampleSize') # DESACTIVER LE WARNING
    for N in range(50,500):
        X = -pi + 2*pi*np.random.rand(N,ninput)
        Y = f(X).reshape((f(X).shape[0],f(X)[0].size))
        tSI,tSIc = rbdfast(Y, X = X)
        SI[:,N-50],SIc[:,N-50] = tSI.reshape((1,3)),tSIc.reshape((1,3))
    #warning('on','RBD:lowSampleSize') #REACTIVER LE WARNING
    
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
    
    #==================== Effect of sample organisation========================
    SIc2 = np.zeros((ninput,450))
    #warning('off','RBD:lowSampleSize') #???????????
    for N in range(50,500):
        X = np.zeros((N,ninput))
        #N+1 values between -pi and +pi
        s0 = np.linspace(-pi,pi,N)
        # 3 random indices for sample size N
        Index = np.matrix([[randint(0,N-1) for z in range(0,ninput)]for n in range(0,N)])
        # Assigning values to the index -> "random" values between [-pi, pi[
        s = np.zeros((N,ninput))
        for line in range(N):
            s[line,:] = s0[Index[line]]
        # Uniform sampling in [0, 1]
        for line in range(N):
            for a in range(ninput):
                X[line,a] = .5 + np.math.asin(np.math.sin(s[line][a]))/pi    
        # Rescaling the uniform sampling between [-pi, pi]
        X = -pi + 2*pi*X        
        Y = f(X).reshape((f(X).shape[0],f(X)[0].size))
        tSIc = rbdfast(Y, Index = Index)[1]
        SIc2[:,N-50] = tSIc.reshape((1,3))
    #warning('on','RBD:lowSampleSize') #???????????????
    
    plt.plot(SIc,'b--')
    plt.plot(SIc2.transpose(),'r--')
    plt.plot([[exactDiag.item(i) for i in range(0,3)] for k in range(50,500)],
               color = '#003366',
               linewidth = 2.0)
    plt.title('Effect of sample organisation')
    plt.ylabel('SI')
    plt.xlabel('Simulation Number')
    plt.show()
    """
    #======================== Effect of M value ===============================
    SIc = np.zeros((ninput,30))
    SI = np.zeros((ninput,30))
    X = -pi + 2*pi*np.random.rand(500,ninput)
    for M in range(1,30):
        SI[:,M],SIc[:,M] = rbdfast(f(X), X = X, M = M)
    
    plt.plot(SIc,'b')
    plt.plot(SI,'r')
    plt.plot([[exactDiag[i] for i in range(0,3)] for k in range(50,500)],'k')
    plt.title('Effect of the M value')
    plt.ylabel('SI')
    plt.xlabel('M value')
    
    print('Tests done')"""
    return

#==============================================================================
#                               RBD FAST
#==============================================================================
def rbdfast(Y, X=np.matrix([]), Index=[], M = 10, test = False):
    """
    RBD python Code for Random Balance Designs
    For the estimation of first order indices
    
       SIc = rbdfast(X,Y) estimation of first order indices according to the X
       values
    
       SIc = rbdfast([],Y,Index) estimation of first order indices according to
       the permutation used to create the sampling design
    
       SIc = rbdfast(X,Y,[],M) estimation of first order indices with a defined
       number of harmonics (default is 10)
    
       X = N-by-k numpy matrix of model inputs
       Y = N-by-l numpy matrix of model output
       Index = N-by-k numpymatrix of permutations for Y to follow
       SIc = k-by-l matrix of estimated first order sensitivity indices unbiased
       SI = k-by-l matrix of estimated first order sensitivity indices biased
    
       N = sample size = total number of model evaluations
       k = number of model input
       l = number of model output
    
    
    Author: S. Tarantola (JRC)
    Joint Research Centre All rights Reserved
    
    Update: M. Rabouille
    Add: Reordering Y according a random design X (EASI algorithm) from E Plischke.
    Add: Unbiased estimator from J-Y Tissot & C Prieur.
    Note: The estimate is less dependant on the M value which can be raised up to 10.
    
    References: 
    S. Tarantola, D. Gatelli and T. Mara (2006)
    Random Balance Designs for the Estimation of First Order 
    Global Sensitivity Indices, Reliability Engineering and System Safety, 91:6, 717-727
    
    Elmar Plischke (2010)
    An effective algorithm for computing global sensitivity indices (EASI)
    Reliability Engineering & System Safety, 95:4, 354-360. <10.1016/j.ress.2009.11.005>
    
    Jean-Yves Tissot, Clémentine Prieur (2012)
    Bias correction for the estimation of sensitivity indices based on random balance designs.
    Reliability Engineering and System Safety, Elsevier,  107, 205-213. <10.1016/j.ress.2012.06.010> <hal-00507526v2>
    """
    # Test function ?? TO DO
    if test:
        return rbdfast_test()

    # Number of harmonics considered for the Fast Fourier Transform must be int
    M = int(M)
    if M <= 0:
        print('Error : M must be a positive integer.')
        return
    
    #If X is not empty, use the Index matrix
    #otherwise, empty Index and use X shape instead
    if X.size == 0:           #.size == 0: #?????
        if Index.size == 0:   #nargin < 3 or isempty(Index):
            #TO DO raise proper error
            print('Error : An index of permutation must be defined.')
            return
        N, k = Index.shape[0], Index[0].size
        useindex = True
    else:
        N, k = X.shape[0], X[0].size
        useindex = False

    if Y.shape[0] != N:
        print('Error : Arguments dimensions are not consistent')
        return

    if N < 2*(M+100):
        print('\n~~~!!!~~~\n',
              'RBD:lowSampleSize\n',
              'Insufficient simulations for proper analysis\n')

    #Initialization of SI and SIc matrices (sensitivity indices)
    SI = np.zeros((k,Y[0].size))
    SIc = np.zeros((k,Y[0].size))
    lamda = (2*M)/N
    Yorg = np.zeros((Y.shape[0],Y[0].size))
    #for every model input in X (or pre-filled Index)
    for col in range(0,k):	
        if useindex:
            # ---- reordering of y wrt a-th index
            for k,ind in enumerate(Index[:,col]):
                Yorg[k,:] = Y[ind,:].copy()
        else:
            # ---- reordering of y wrt a-th variable
            #Get permutation ordering X wrt a-th variable
            sortingPerm = sorted(range(len(X[:,col])), key = lambda k:X[:,col][k])
            #Order the permutation as to form a saw profile
            finalPerm = sortingPerm[0::2]+sortingPerm[-1-N%2::-2]
            #copy ordered Y into new Yorg variable
            for i,ind in enumerate(finalPerm):
                Yorg[i,:] = Y[ind,:].copy()

        #-----calculation spe1 at integer frequency
        spectrum = np.abs(np.fft.rfft(Yorg[:,0]))**2/N/(N-1)        #(abs(fft(Yorg)))^2/N/(N-1)
        # Normalization by N-1 to match the definition of the unbiased variance
        # We thus have the same definition as var(Y)
        # var(Yorg) ~ sum(spectrum(2:end,:))
        # var(Yorg|Xi) ~ 2*sum(spectrum(2:M+1,:)) = sum(spectrum(2:M+1,:))+sum(spectrum(end:-1:(end+1-M),:))
    
        V = sum(spectrum[1:])                         #sum(spectrum(2:end,:))
        SI[col,:] = sum(spectrum[1:M+1])/V #facteur 2 inutile???
        SIc[col,:] = SI[col,:] - lamda/(1-lamda)*(1-SI[col,:])
    
    return SI,SIc




#==============================================================================
#           Bootstrap indicator
#==============================================================================
"""
????????

if isfield(analyse,'bootstrap') &&  analyse.bootstrap
	fprintf('-> Bootstrap analysis.\n')
	for rep=1:analyse.bootstrap_param.rep
		# Selection aléatoire de données avec retirage
		Ind = randi(nbs_sorties,analyse.bootstrap_param.ech,1);
		
		if params.type_ech==3
			SIbs_rbdfast(:,:,rep)=rbd_fast(1,true,analyse.RBD.harmonics,params.index_rbd_fast(Ind,:),resultat.sorties(Ind,:));
		else
			SIbs_rbdfast(:,:,rep)=rbd_fast(1,true,analyse.RBD.harmonics,[],resultat.sorties(Ind,:),params.plan(Ind,1:params.variables.vars_nb));
		end
	end
	analyse.SIbs_rbdfast_mean = mean(SIbs_rbdfast,3);
	analyse.SIbs_rbdfast_var = var(SIbs_rbdfast,0,3);
end
"""