# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:53:15 2017

@author: Sarah Juricic
"""

import numpy as np
warning_on = False


def to_integer(m):
    """
    m : float or int, the number of harmonics considered in the SA
    returns int, floor of M if M is float
    Raises an error if not positive
    """

    m = int(m)

    if m <= 0:
        raise ValueError('m must be a positive integer.')

    return m


def get_sorting_permutation(array):
    """
    gets the triangle sorting permutation
    x : single column array
    return perm, single column array of permutation
    """

    length = array.shape[0]

    # get asceding permutation of X
    ascending_perm = sorted(range(len(array)), key=lambda k: array[k])

    # Order the permutation as to form a triangle profile
    perm = ascending_perm[0::2] + ascending_perm[-1 - length % 2::-2]

    return perm


def bootstrap(k, l, x, y):
    """
    Bootstrap function to evaluate sensitivity of RBD-Fast estimates
    1000 sample
    Takes as arguments :
    x model inputs
    y model outputs
    k number of model input
    l number of model output
    return mean, var
    """

    sample_size = Y.shape[0]
    all_si_c = np.empty((1000, k, l))

    # calculate sensitivity coef 1000 times
    for i in range(0, 1000):
        # random sample on sample_size n with replacement
        indices = np.random.randint(0, high=sample_size, size=sample_size)
        x_new = np.empty((sample_size, k))
        y_new = np.empty((sample_size, l))
        for j, index in enumerate(indices):
            x_new[j, :] = x[index, :]
            y_new[j, :] = y[index, :]
        all_si_c[i] = rbdfast(y_new, x_new)[1]

    return np.mean(all_si_c, axis=0), np.var(all_si_c, axis=0)


#==============================================================================
#                               RBD FAST
#==============================================================================
def rbdfast(y, x=np.array([]), index=np.array([]),
            m=10, bootstrap=False, warning_on=True):
    """
    RBD python Code for Random Balance Designs
    For the estimation of first order indices

       si_c = rbdfast(x,y) estimation of first order indices according to the x
       values

       si_c = rbdfast([],y,index) estimation of first order indices according to
       the permutation used to create the sampling design

       si_c = rbdfast(x,y,[],m) estimation of first order indices with a defined
       number of harmonics (default is 10)

       x = n-by-k numpy matrix of model inputs
       y = n-by-l numpy matrix of model output
       CAUTION : if number of lines != n, this WILL end up in error
       Index = n-by-k numpymatrix of permutations for y to follow
       m : integer < n, threshold for the calculation of the indices
       si_c = k-by-l matrix of estimated first order sensitivity indices unbiased
       si = k-by-l matrix of estimated first order sensitivity indices biased

       n = sample size = total number of model evaluations
       k = number of model input
       l = number of model output

    """
    # Number of harmonics considered for the Fast Fourier Transform must be int
    m = to_integer(m)

    # If X is not empty, use the Index matrix
    # otherwise, empty Index and use X shape instead
    if x.size == 0:  # .size == 0: #?????
        if index.size == 0:  # nargin < 3 or isempty(Index):
            # TO DO raise proper error
            print('Error : An index of permutation must be defined.')
            return
        n, k = index.shape[0], index[0].size
        useindex = True
    else:
        n, k = x.shape[0], x[0].size
        useindex = False

    if y.shape[0] != n:
        print('Error : Arguments dimensions are not consistent')
        return

    if n < 2 * (m + 100):
        warning_on = True or warning_on

    # Initialization of SI and SIc matrices (sensitivity indices)
    si = np.zeros((k, y[0].size))
    si_c = np.zeros((k, y[0].size))
    lamda = (2 * m) / n
    y_org = np.zeros((y.shape[0], y[0].size))
    # for every model input in x (or pre-filled Index)
    for col in range(0, k):
        if useindex:
            # ---- reordering of y wrt a-th index
            for k, ind in enumerate(index[:, col]):
                y_org[k, :] = y[ind, :].copy()
        else:
            # ---- reordering of y wrt a-th variable of x
            permut = get_sorting_permutation(x[:, col])
            # copy ordered Y into new Yorg variable
            for i, ind in enumerate(permut):
                y_org[i, :] = y[ind, :].copy()

        # calculate spectrum with RFFT : (abs(fft(Yorg)))^2/N/(N-1)
        # Normalization by N-1 to match the definition of the unbiased variance
        spectrum = np.abs(np.fft.rfft(y_org[:, 0]))**2 / n / (n - 1)
        v = sum(spectrum[1:])
        si[col, :] = sum(spectrum[1:m + 1]) / v
        si_c[col, :] = si[col, :] - lamda / (1 - lamda) * (1 - si[col, :])

        #optionnal bootstrap analysis
        if bootstrap == True:
            print('Bootstrap analysis :\n')
            bootstrap()

#        if warning_on:
#            print('\n~~~!!!~~~\n',
#                  'There has been at least on low Sample Size\n',
#                  'Insufficient simulations for proper analysis\n')
    return si, si_c



#==============================================================================
#           Bootstrap indicator
#==============================================================================
"""
????????

if isfield(analyse,'bootstrap') &&  analyse.bootstrap
	fprintf('-> Bootstrap analysis.\n')
	for rep = 1:analyse.bootstrap_param.rep
		# Selection aléatoire de données avec retirage
		Ind = randi(nbs_sorties,analyse.bootstrap_param.ech,1);

		if params.type_ech==3
			SIbs_rbdfast(:,:,rep)=rbd_fast(1,true,analyse.RBD.harmonics,params.index_rbd_fast(Ind,:),resultat.sorties(Ind,:));
		else
			SIbs_rbdfast(:,:,rep)=rbd_fast(1,true,analyse.RBD.harmonics,[],resultat.sorties(Ind,:),params.plan(Ind,1:params.variables.vars_nb));

	analyse.SIbs_rbdfast_mean = mean(SIbs_rbdfast,3);
	analyse.SIbs_rbdfast_var = var(SIbs_rbdfast,0,3);
end
"""
