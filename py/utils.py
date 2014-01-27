import numpy as np
import random
import pyfits as pyf
import matplotlib
import matplotlib.pyplot as plt
import random
import triangle

def make_triangle_plot(sampler, labels):
    trichain = np.column_stack((sampler.flatchain, sampler.lnprobability.flatten()))
    return triangle.corner(trichain, labels=labels)

def get_best_lnprob(sampler):
    highln = np.argmax(sampler.lnprobability)
    return sampler.flatchain[highln], sampler.lnprobability.flatten()[highln]

def make_walker_plots(sampler, labels, nwalkers):
    plots = []
    for j,par in enumerate(labels):
        fig = plt.figure()
        axis = fig.gca()
        if par == "ln_prob":
            for i in range(nwalkers):
                axis.plot(sampler.lnprobability[i, :])
        else:
            for i in range(nwalkers):
                axis.plot(sampler.chain[i, :, j])
        axis.set_xlabel('Step Number')
        axis.set_ylabel('{}'.format(par))
        plots.append(fig)
    return plots

            


def make_band_time_grid(inittime, finaltime, dt, bandlist):
    """
    Inputs: inittime: start of grid
            finaltime: end of grid
            dt: spacing
            bandlist: list of bands in data
    Outputs: fulltimegrid: grid of times, repeated 5 times
             fullbands: bands matching each time point
    """
    fulltimegrid = []
    fullbands = []
    for i, name in enumerate(bandlist):
        timegrid = np.arange(0.5 * dt + inittime, finaltime, dt)
        bands = [i] * len(timegrid)
        fulltimegrid.extend(timegrid)
        fullbands.extend(bands)

    return np.array(fulltimegrid), np.array(fullbands)


def ln_1d_gauss(x, m, sigmasqr):
    """
    Compute log of 1d Gaussian
    Inputs: x is the variable
            m is the mean
            sigmasqr is the variance (squared)
    Output: Natural log of the gaussian evalulated at x
    """
    A = 1. / np.sqrt(sigmasqr * 2 * np.pi)
    return np.log(A)-(1 / 2.) * (x - m) ** 2 / sigmasqr


def ln_likelihood(r, sr2, mean_r, sr2_meanr):
    """
    Inputs: r is the r magnitude
            sr2 is the variance of r
            mean_r is the mean of r
            sr2_meanr is he variance of all r
    Output: Log-likelihood
    """
    return ln_1d_gauss(r, mean_r, sr2 + sr2_meanr)


def tot_likelihood(params, r, sr2):
    mean_r, sr2_meanr = params
    return np.sum(ln_likelihood(r, sr2, mean_r, sr2_meanr))


def make_2d_grid(min1, max1, min2, max2, step1, step2):
    """
    Makes a 2d grid for searching
    Inputs: min1: minimum of parameter 1
            max1: maximum of parameter 1
            min2: minimum of parameter 2
            max2: maximum of parameter 2
            step1: spacing of parameter 1
            step2: spacing of parameter 2
    """
    grid1 = np.arange(min1 + .5 * step1, max1, step1)
    grid2 = np.arange(min2 + .5 * step2, max2, step2)
    return np.meshgrid(grid1, grid2), grid1, grid2


def grid_search_var_mean(mag, mag_err, grid=None):
    """
    Inputs:
    mag: magnitudes
    mag_err: sigma squared values
    grid (optional): search grid
    Outputs:
    max_mag is the mean for which the probability is max
    std is the square root of the variance where the probability is max
    """

    if grid:
        [mm, ss], means, sigmas, = grid
    else:
        [mm, ss], means, sigmas = make_2d_grid(10., 30., 0., 1., 0.1, .001)
    probs = np.zeros_like(mm)

    #do stupid loop
    ni, nj = mm.shape
    for i in range(ni):
        for j in range(nj):
            pars = np.array([mm[i, j], ss[i, j]])
            probs[i, j] = tot_likelihood(pars, mag, mag_err)
    loc = np.argmax(probs)
    sigloc, meanloc = np.unravel_index(loc, probs.shape)
    max_mag = means[meanloc]
    std = np.sqrt(sigmas[sigloc])

    return max_mag, std


def grid_search_all_bands(mag, mag_err, bandsnum, grid=None):
    """
    Inputs:
    mag: magnitudes
    mag_err: sigma squared values
    bandsnum: bands matching the mag list
    grid (optional): search grid
    Outputs:
    means are the means for which the probability is max in each band
    amps are the square root of the variance
         where the probability is max in each band
    """

    means = []
    amps = []
    for i in np.sort(np.unique(bandsnum)):
        mask = [bandsnum == i]
        mean, a = grid_search_var_mean(mag[mask], mag_err[mask], grid)
        means.append(mean)
        amps.append(a)
    return np.array(means), np.array(amps)


def grid_search_obj_band(obj, band, grid=None):
    """
    Inputs:
    obj - data object
    band - bandname we are interested in
    grid (optional): search grid
    Outputs:
    max_mag is the mean for which the probability is max
    std is the square root of the variance where the probability is max
    """

    return grid_search_var_mean(obj.get_mags(band),
                                obj.get_sigmas(band, obj) ** 2, grid)


def grid_search_obj_all(obj, grid=None):
    """
    Inputs:
    obj - data object
    grid (optional): search grid
    Outputs:
    means are the means for which the probability is max in each band
    amps are the square root of the variance
         where the probability is max in each band
    """

    means = []
    amps = []
    for name in obj.get_bandnames:
        mean, a = grid_search_obj_band(obj.get_mags(name),
                                       obj.get_sigmas(name) ** 2, grid)
        means.append(mean)
        amps.append(a)
    return np.array(means), np.array(amps)


def make_prior_plots(quasarobj, timegrid, bands, pmean, psig, means, bandslist,
                     num_samps=8):
    """
    Inputs: quasarobj: obj for the quasar we care about
            timegrid: grid of sample points
            bands: grid of bands matching the timegrid
            pmean: the mean at each point matching the timegrid
            psig: the sigma at each point matching the timegrid
            means: the overall means in each band
            num_samps (optional) number of samples to plot
    Note: should probably return a figure object at some point
           but i have not written that yet
    """
    maggrids = []
    plt.subplots_adjust(hspace=0, top=.95)
    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)
    for i in range(num_samps):
        maggrid = quasarobj.get_prior_sample(timegrid, bands)
        maggrid = np.array(maggrid).reshape(timegrid.shape)
        maggrids.append(maggrid)

    for i in range(5):
        ax = plt.subplot(511 + i)
        mask = [bands == i]

        btimegrid = timegrid[mask]
        bpmean = pmean[mask]
        bpsig = psig[mask]
        plt.plot(btimegrid, bpmean, 'k-')
        plt.plot(btimegrid, bpmean-bpsig, 'k-')
        plt.plot(btimegrid, bpmean+bpsig, 'k-')
        for j in range(num_samps):
            maggrid = maggrids[j]
            plt.plot(btimegrid, maggrid[mask], 'k-', alpha=0.25)
        plt.setp(ax.get_xticklabels(), visible=(i == 4))
        plt.ylabel('%s' % bandslist[i])
        plt.ylim(means[i] - .75, means[i] + .75)


def make_posterior_plots(quasarobj, times, mags, bands, sigmas, timegrid,
                         bandsgrid, pmean, psig, means, bandlist, num_samps=8):
    """
    Inputs: quasarobj: obj for the quasar we care about
            times: measured times
            mags: measured mags
            bands: bands matching each mag
            sigmas: measured sigmas
            timegrid: grid of sample points
            bands: grid of bands matching the timegrid
            pmean: the mean at each point matching the timegrid
            psig: the sigma at each point matching the timegrid
            means: the overall means in each band
            num_samps (optional) number of samples to plot
    Note: should probably return a figure object at some point
           but i have not written that yet
    """

    maggrids = []
    plt.subplots_adjust(hspace=0, top=.95)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)

    for i in range(num_samps):
        maggrid = quasarobj.get_conditional_sample(timegrid, bandsgrid,
                                                   mags, times, bands,
                                                   sigmas)
        maggrid = np.array(maggrid).reshape(timegrid.shape)
        maggrids.append(maggrid)

    for i in range(5):
        ax = plt.subplot(511 + i)
        maskgrid = [bandsgrid == i]
        mask = [bands == i]
        plt.ylabel('%s' % bandlist[i])
        plt.errorbar(times[mask], mags[mask], yerr=sigmas[mask],
                     linestyle='none', color='black', marker='.')
        btimegrid = timegrid[maskgrid]
        bpmean = pmean[maskgrid]
        bpsig = psig[maskgrid]
        plt.plot(btimegrid, bpmean, 'k-')
        plt.plot(btimegrid, bpmean-bpsig, 'k-')
        plt.plot(btimegrid, bpmean+bpsig, 'k-')
        for j in range(num_samps):
            maggrid = maggrids[j]
            plt.plot(btimegrid, maggrid[maskgrid], 'k-', alpha=0.25)
        plt.setp(ax.get_xticklabels(), visible=(i == 4))
        plt.ylim(means[i] - .75, means[i] + .75)

def make_data_plots(obj,prefix='quasar_data'):
    plt.clf()
    plt.subplots_adjust(hspace=0, top=.95)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)

    mags, times, bands, sigmas = obj.get_data()
    start = np.min(times)-50.
    end = np.max(times)+50.
    bandlist = obj.get_bandlist()

    for i in range(5):
        ax = plt.subplot(511+i)
        mask = [bands==i]
        plt.xlim(start,end)
        plt.ylabel('%s' % bandlist[i])
        plt.errorbar(times[mask], mags[mask], yerr=sigmas[mask], linestyle='none', color='black', marker='.')
        plt.setp(ax.get_xticklabels(),visible=(i == 4))

    plt.savefig('{}.png'.format(prefix))


def mock_panstarrs(obj, time_step):
    """
    Inputs: obj: data object
            time_step: slicing interval
    Returns:
             Sampled mags, times, sigmas, bands
    """
    times = obj.get_times()
    start_time = np.min(times)
    end_time = np.max(times)
    mags = obj.get_mags()
    sigmas = obj.get_sigmas()
    bands = obj.get_bands()

    new_mags = []
    new_times = []
    new_sigmas = []
    new_bands = []

    for i in np.arange(start_time, end_time, time_step):
        mask = [(times >= i) & (times < (i + time_step))]
        if not np.any(mask):
            continue
        mask_mags = mags[mask]
        mask_times = times[mask]
        mask_sigmas = sigmas[mask]
        mask_bands = bands[mask]
        samp = np.random.randint(len(mask_mags))
        new_mags.append(mask_mags[samp])
        new_times.append(mask_times[samp])
        new_sigmas.append(mask_sigmas[samp])
        new_bands.append(mask_bands[samp])
    return new_mags, new_times, new_sigmas, new_bands


def plot_meshgridmeanmin(meanmin, maxmin, sigmamin, sigmamax, meanstep, sigmastep, band, obj):
    """
    Inputs: see function above
    Outputs: plots mesh grid
    THIS DOESNT WORK ANYMORE
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
    sigmas = np.arange(sigmamin, sigmamax, sigmastep)
    means = np.arange(meanmin, meanmax, meanstep)
    mm, ss = np.meshgrid(means, sigmas)
    probs = np.zeros_like(mm)

    #do stupid loop
    ni, nj = mm.shape
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
    NOTE: DOESNT WORK ANYMORE
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
    """
    NOTE: NO LONGER WORKS
    """
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
    NOTE: NO LONGER WORKS
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
