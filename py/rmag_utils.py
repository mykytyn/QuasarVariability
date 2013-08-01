import numpy as np
import random

def ln_1d_gauss(x,m,sigmasqr):
    """
    Compute log of 1d Gaussian
    Inputs: x is the variable
            m is the mean
            sigmasqr is the variance (squared)
    Output: Natural log of the gaussian evalulated at x
    """
    A = 1./np.sqrt(sigmasqr*2*np.pi)
    return np.log(A)-(1/2.)*(x-m)**2/(sigmasqr)

def ln_likelihood(r,sr2,mean_r,sr2_meanr):
    """
    Inputs: r is the r magnitude
            sr2 is the variance of r
            mean_r is the mean of r
            sr2_meanr is he variance of all r
    Output: Log-likelihood
    """
    return ln_1d_gauss(r,mean_r,sr2+sr2_meanr)

def tot_likelihood(params, r, sr2):
    mean_r,sr2_meanr=params
    return np.sum(ln_likelihood(r,sr2,mean_r,sr2_meanr))
