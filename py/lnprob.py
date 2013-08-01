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

def ln_like_array(tmag,gmag,g_mean,st2,sg2,a,b,s2):
    """
    Inputs: tmag is the target magnitude
            gmag is the given magnitude
            g_mean is the mean of the given magnitude
            st2 is the variance of the target magnitude
            sg2 is the variance of the given magnitude
            a is a scale factor (power)
            b is an offset
            s2 is a jitter in variance units
    Output: Log-likelihood
    """
    return ln_1d_gauss(tmag,a*(gmag-g_mean)+b,a*a*sg2+st2+s2)

def ln_prob(params,t,g,st,sg):
    """
    Params:
           a is a scale factor (power)
           b is an offset
           s is a jitter in std. dev units
           g_mean is the mean of the given magnitude
    Data:
           t is the target magnitude
           g is the given magnitude
           st is the error of the target magnitude
           sg is the error of the target magnitude
    """
    a,b,s,g_mean = params

    return np.sum(ln_like_array(t,g,g_mean,st**2,sg**2,a,b,s**2))
