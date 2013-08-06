import numpy as np
import random
import pyfits as pyf
import matplotlib.pyplot as plt
from QuasarVariability import QuasarVariability
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

def opentable(fn,obj):
    """
    Inputs: prefix of fits file name, obj headojbid
    Outputs: table data for given obj
    """
    data=pyf.open('%s.fits' %(fn))[1].data
    mask = data['headobjid']==obj
    table = data[mask]
    return table

def get_mag(band,obj):
    """
    Input: ugriz band and headobjid
    Output: magnitude data for object in given band
    """
    table=opentable('quasar',obj)
    mag=table['psfMag_%s'%(band)]
    return mag

def get_mag_err(band,obj):
    """
    Input: ugriz band and headobjid
    Output: errors  in magnitude for given band and obj
    """
    table=opentable('quasar',obj)
    mag_err=table['psfMagerr_%s'%(band)]
    return mag_err

def get_time(band,obj):
    """
    Input: ugriz band and headobjid
    Output: mjd times for given band and obj
    """
    table=opentable('quasar',obj)
    return list(table['mjd_%s'%(band)])

def meshgrid(meanmin,maxmin, sigmamin, sigmamax, meanstep, sigmastep, band, obj):
    """
    Inputs:
    meanmin is the minimum range of the mean 
    meanmax is the maximum range of the mean
    sigmamin is the minimum range of the variance
    sigmamax is the maximum range of the variance
    meanstep is the step size for means
    sigmastep is the step size for variance
    band is the ugriz band to be observed
    obj is the headobjid 
    Outputs:
    max_mag is the mean for which the probability is max
    std is the square root of the variance where the probability is max
    """
    # set up arrays
    meanmin = meanmin
    meanmax = maxmin
    sigmamin = sigmamin
    sigmamax = sigmamax
    meanstep = meanstep
    sigmastep = sigmastep
    sigmas = np.arange(sigmamin,sigmamax,sigmastep)
    means = np.arange(meanmin,meanmax,meanstep)
    mm,ss = np.meshgrid(means,sigmas)
    probs = np.zeros_like(mm)

    #do stupid loop
    ni,nj = mm.shape
    print ni,nj
    mag=get_mag(band,obj)
    mag_err=get_mag_err(band,obj)**2
    for i in range(ni):
        for j in range(nj):
            pars = np.array([mm[i,j],ss[i,j]])
            probs[i,j] = tot_likelihood(pars,mag, mag_err)
    #print probs
    loc=np.argmax(probs)
    sigloc,meanloc=np.unravel_index(loc,probs.shape)
    max_mag=means[meanloc]
    std=np.sqrt(sigmas[sigloc])
    
    return max_mag, std

def plot_meshgridmeanmin(meanmin,maxmin, sigmamin, sigmamax, meanstep, sigmastep, band, obj):
    """
    Inputs: see function above
    Outputs: plots mesh grid
    """
    plt.figure()
    plt.clf()
    # set up arrays
    meanmin = meanmin
    meanmax = maxmin
    sigmamin = sigmamin
    sigmamax = sigmamax
    meanstep = meanstep
    sigmastep = sigmastep
    sigmas = np.arange(sigmamin,sigmamax,sigmastep)
    means = np.arange(meanmin,meanmax,meanstep)
    mm,ss = np.meshgrid(means,sigmas)
    probs = np.zeros_like(mm)

    #do stupid loop
    ni,nj = mm.shape
    print ni,nj
    mag=get_mag(band,obj)
    mag_err=get_mag_err(band,obj)**2
    for i in range(ni):
        for j in range(nj):
            pars = np.array([mm[i,j],ss[i,j]])
            probs[i,j] = tot_likelihood(pars,mag, mag_err)
    print probs

    vmax = np.max(probs)
    vmin = vmax-1000.

    plt.imshow(probs,cmap='gray',interpolation='nearest',origin='lower',
               extent=[meanmin,meanmax,sigmamin,sigmamax+.05], vmax=vmax,vmin=vmin,aspect='auto')
    plt.colorbar()
    plt.savefig('grid_%s_%s.png' %(band,obj))
    return 'finished plotting'

def plot_mag(band,obj):
    """
    Input:
    band is the ugriz band 
    obj is the headobjid
    Output: plot for the magnitude in this band with a solid line at the mean magnitude and fainter lines showin    g the 1sigma value from the mean magnitude
    """
    mag=get_mag(band,obj)
    mag_err=get_mag_err(band,obj)
    time=get_time(band,obj)
    print sorted(time),len(time)
    plt.errorbar(time, mag, yerr=mag_err, marker='.',color='black', ls='none', label='%s'%(band))
    band_lines=meshgrid(10., 30., 0., 1., .1, .001, band, obj)
    plt.axhline(y=band_lines[0],color='black',alpha=0.5,linewidth=2)
    plt.axhline(y=band_lines[0]+band_lines[1],color='black',linewidth=.1,alpha=0.5)
    plt.axhline(y=band_lines[0]-band_lines[1], color='black',linewidth=.1,alpha=0.5)
    plt.ylim(band_lines[0]-0.75,band_lines[0]+0.75)
    plt.ylabel('%s'%(band))
    #plt.legend(loc='upper left',prop={'size':8})
    
def plot_mag_curve(band,obj):
    mean,a=meshgrid(10., 30., 0., 1., .1, .001, band, obj)
    tau = 200.
    testQ= QuasarVariability(a,tau,mean)
    timegrid=np.arange(51000,54500,20)
    testMags=get_mag(band,obj)
    testTimes=get_time(band,obj)
    testSigmas=get_mag_err(band,obj)
    pmean,Vpp=testQ.get_conditional_mean_and_variance(timegrid,testMags,testTimes,testSigmas)
    pmean=np.array(pmean).reshape(timegrid.shape)
    psig=np.sqrt(np.diag(np.array(Vpp)))
    plt.errorbar(testTimes, testMags, yerr=testSigmas, marker='.',color='black', ls='none', label='%s'%(band))
    plt.plot(timegrid,pmean,color='black',alpha=0.5,linewidth=2)
    plt.plot(timegrid,pmean+psig,color='black',linewidth=.1,alpha=0.5)
    plt.plot(timegrid,pmean-psig, color='black',linewidth=.1,alpha=0.5)
    plt.ylim(mean-0.75,mean+0.75)
    plt.ylabel('%s'%(band))
    for i in range(4):
        maggrid = testQ.get_conditional_sample(timegrid,testMags,testTimes,testSigmas)
        maggrid = np.array(maggrid).reshape(timegrid.shape)
        plt.plot(timegrid,maggrid,'k-',alpha=0.25)

def missing_points(band,obj):
    """
    Inputs: 
    band is ugriz band
    obj is headobjid
    Output: faint gray triangles for data that is outside of the limits of magnite range
    """
    mag=get_mag(band,obj)
    time=get_time(band,obj)
    band_lines=meshgrid(10., 30., 0., 1., .1, .001, band, obj)
    ymin, ymax= band_lines[0]-0.75, band_lines[0]+0.75
    for m,t in zip(mag,time):
        if m > ymax:
            print m
            plt.plot(t, ymax-0.05, marker='^',markerfacecolor='gray', mew=0,alpha=0.5)
        if m < ymin:
            print m
            plt.plot(t,ymin+0.05, marker='v',markerfacecolor='gray', mew=0,alpha=0.5)

def mock_panstarrs(obj,time_step):
    mjd=get_time('u',obj)
    keep_this_data={}
    print mjd[0]
    for i in np.arange(sorted(mjd)[0],sorted(mjd)[len(mjd)-1],time_step):
        bands={'u': 0, 'g': 1, 'r': 2, 'i':3, 'z':4}
        data_per_interval={}
        for band in bands:
            rand_time=get_time(band,obj)
            for time in rand_time:
                if i < time < i+time_step:
                    #print i, i+time_step, time
                    data_per_interval[time]=band
            else:
                print i, i+time_step, 'no data'
        if len(data_per_interval) > 0:
            print data_per_interval.keys(), len(data_per_interval)
            data_time=random.choice(data_per_interval.keys())
            #data_time=data_per_interval.keys()[data_time]
            data_band=data_per_interval[data_time]
            index=get_time(data_band,obj).index(data_time)
            data_mag=get_mag(data_band, obj)[index]
            data_magerr=get_mag_err(data_band,obj)[index]
            values=data_time,data_mag,data_magerr
            print values
            keep_this_data[values]=data_band

    return keep_this_data
    
