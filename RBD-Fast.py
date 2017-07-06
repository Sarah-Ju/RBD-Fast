# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:53:15 2017

With the agreement of Mickael Rabouille, author of the original Matlab code

@author: translation to python 3.5 Sarah Juricic
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class IsNotIntegerError(Exception):
    pass

def to_integer(M):
    """
    M : float, the number of harmonics considered in the SA
    returns int, floor of M if M is float
    Raises an error if not positive
    """
    M = int(M)
    if M <= 0:
        raise IsNotPositiveIntegerError
    return M

def get_sorting_permutation(X):
    """
    gets the triangle sorting permutation
    X : single column array
    return perm, single column array of permutation
    """
    #get asceding permutation of X
    ascending_perm = sorted(range(len(X)), key = lambda k:X[k])
    #Order the permutation as to form a triangle profile
    perm = ascending_perm[0::2] + ascending_perm[-1-N%2::-2]
    return perm

#==============================================================================
#                               RBD FAST
#==============================================================================
def rbdfast(Y, X=np.matrix([]), Index=[], M = 10):
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
    # Number of harmonics considered for the Fast Fourier Transform must be int
    M = to_integer(M)
    
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