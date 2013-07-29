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

def ln_like_array(g,r,sr2,sg2,a,b,s2):
    """
    Inputs: g is the g magnitude
            r is the r magnitude
            sr2 is the variance of r
            sg2 is the variance of g
            a is a scale factor (power)
            b is an offset
            s2 is a jitter in variance units
    Output: Log-likelihood
    """
    return ln_1d_gauss(g,a*r+b,sr2+sg2+s2)

def ln_prob(params,g,r,sg,sr):
    a,b,s = params
    #g,r,sg,sg = data

    return np.sum(ln_like_array(g,r,sr**2,sg**2,a,b,s**2))
