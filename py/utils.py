import numpy as np
import pyfits as pyf
import matplotlib
import matplotlib.pyplot as plt
import triangle



def make_triangle_plot(lnprobability, flatchain, labels):
    """
    NOTES: MAKE THIS NOT MAGIC
    """
    lnprob = lnprobability.flatten()
    trichain = np.column_stack((flatchain, lnprob))
    print trichain[:,5:].shape
    extents = list([[np.min(x[np.isfinite(x)]), np.max(x[np.isfinite(x)])] for
                    x in np.hsplit(trichain, trichain.shape[1])])
    extents.append([np.min(lnprob[np.isfinite(lnprob)]), np.max(lnprob)])
    return triangle.corner(trichain[:,5:], labels=labels[5:], extents=extents[5:])


def get_best_lnprob(sampler):
    """
    Gets the highest lnprob from a sampler chain
    TODO: make it not depend on an actual sampler object
    """
    highln = np.argmax(sampler.lnprobability)
    return sampler.flatchain[highln], sampler.lnprobability.flatten()[highln]


def make_walker_plots(sampler, labels, nwalkers):
    """
    Creates MCMC walker plots for each parameter
    TODO: make not depend on an actual sampler object
    """
    plots = []
    for j, par in enumerate(labels):
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

#I think all of these functions will be replaced by a general optimizer soon - might even be incorrect
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


def plot_points(ax, mags, times, errors, colors, median):
    #put in **kwargs
    for m, t, e, c in zip(mags, times, errors, colors):
        if np.abs(m-median)<.75: #MAGIC
            ax.errorbar(t, m, yerr=e, ecolor=c, linestyle='none', color=c, marker='.', alpha=.75)
        elif m-median>.75:
            ax.errorbar(t,median+.7, ecolor=c, linestyle='none', color=c, marker='^', alpha=.75)
        else:
            ax.errorbar(t,median-.7, ecolor=c, linestyle='none', color=c, marker='v', alpha=.75)
    return ax

def make_prior_plots(quasarobj, timegrid, bands, pmean, psig, means, bandslist,
                     num_samps=8):
    """
    I BROKE THIS SOMEHOW SORRY WILL FIX SORRY
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
    plt.clf()
    pmean = quasar.get_mean_vector(timegrid, bandsgrid)
    Vpp = quasar.get_variance_tensor(timegrid, bandsgrid)
    pmean = np.array(pmean).reshape(timegrid.shape)
    psig = np.sqrt(np.diag(np.array(Vpp)))
    fig = plt.figure()
    fig.subplots_adjust(hspace=0, top=.95)
    maggrids = []
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    for i in range(num_samps):
        maggrid = quasarobj.get_prior_sample(timegrid, bands)
        maggrid = np.array(maggrid).reshape(timegrid.shape)
        maggrids.append(maggrid)

    for i in range(5):
        ax = fig.add_subplot(511 + i)
        mask = [bands == i]
        ax.label_outer()
        btimegrid = timegrid[mask]
        bpmean = pmean[mask]
        bpsig = psig[mask]
        ax.plot(btimegrid, bpmean, 'k-')
        ax.plot(btimegrid, bpmean-bpsig, 'k-')
        ax.plot(btimegrid, bpmean+bpsig, 'k-')
        for j in range(num_samps):
            maggrid = maggrids[j]
            ax.plot(btimegrid, maggrid[mask], 'k-', alpha=0.25)
        ax.set_ylabel('%s' % bandslist[i])
        ax.set_ylim(means[i] - .75, means[i] + .75) #MAGIC
    return fig

def make_posterior_plots(quasar, quasar_data, deltalns=None, num_samps=8):
    """
    Inputs: quasar: obj for the quasar we care about
            quasar_data: data for quasar
            deltalns: changes to lnprob for each set of points
            num_samps (optional) number of samples to plot
    """

    bands_dict = quasar_data.get_banddict()
    totalmags, totaltimes, totalbands, totalsigmas, totalbad = quasar_data.get_data()
    bandsname = quasar_data.get_bandnames()
    bandlist = quasar_data.get_bandlist()
    medians = []
    dt = 1.0
    initial_time = np.min(totaltimes)-25
    final_time = np.max(totaltimes)+25
    timegrid, bandsgrid = make_band_time_grid(initial_time,
                                              final_time, dt,
                                              bandlist)

    for i in range(5):
        mask = [totalbands == i]
        medians.append(np.median(totalmags[mask]))

    pmean, Vpp = quasar.get_conditional_mean_and_variance(timegrid, bandsgrid,
                                                          totalmags, totaltimes,
                                                          totalbands, totalsigmas)
    pmean = np.array(pmean).reshape(timegrid.shape)
    psig = np.sqrt(np.diag(np.array(Vpp)))


    fig = plt.figure()
    fig.subplots_adjust(hspace=0, top=.95)
    maggrids = []
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    if deltalns is not None:
        mindeltalns = np.min(deltalns)
        deltalns = deltalns-mindeltalns
        maxdeltalns = np.max(deltalns)

    for i in range(num_samps):
        maggrid = quasar.get_conditional_sample(timegrid, bandsgrid,
                                                   totalmags, totaltimes, totalbands,
                                                   totalsigmas)
        maggrid = np.array(maggrid).reshape(timegrid.shape)
        maggrids.append(maggrid)

    for i in range(5):
        ax = fig.add_subplot(511 + i)
        ax.label_outer()
        ax.set_xlim(initial_time, final_time)
        ax.set_ylabel('%s' % bandlist[i])
        maskgrid = [bandsgrid == i]
        mags, times, bands, sigmas, bad = quasar_data.get_data(bandname = bandlist[i])
        if deltalns is not None:
            colors = [str(1-x/maxdeltalns) for x in deltalns]
        else:
            colors = ['black']*len(mags)
        plot_points(ax, mags, times, sigmas, colors, medians[i])
        btimegrid = timegrid[maskgrid]
        bpmean = pmean[maskgrid]
        bpsig = psig[maskgrid]
        ax.plot(btimegrid, bpmean, 'k-', lw=.5)
        ax.plot(btimegrid, bpmean-bpsig, 'k-', lw=.5, alpha=.5)
        ax.plot(btimegrid, bpmean+bpsig, 'k-', lw=.5, alpha=.5)
        for j in range(num_samps):
            maggrid = maggrids[j]
            ax.plot(btimegrid, maggrid[maskgrid], 'k-', alpha=0.25)

        ax.set_ylim(medians[i] - .75, medians[i] + .75)
    return fig

def make_data_plots(quasar_data):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0, top=.95)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)

    times = quasar_data.get_times()
    start = np.min(times)-50.
    end = np.max(times)+50.
    bandlist = quasar_data.get_bandlist()

    for i in range(5):
        mags, times, bands, sigmas, bad = quasar_data.get_data(bandname = bandlist[i])
        ax = fig.add_subplot(511 + i)
        ax.label_outer()
        ax.set_xlim(start, end)
        ax.set_ylabel('%s' % bandlist[i])
        median = np.median(mags)
        colors = ['black']*len(mags)
        plot_points(ax, mags, times, sigmas, colors, median)
        ax.set_ylim(median-.75,median+.75)

    return fig


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


def calc_delta_ln(bestparams, quasar_data, default, onofflist):
    """
    Might not be needed anymore, might not work, do not use
    """
    oldtimes = quasar_data.get_times(bandname='r')
    oldbad = quasar_data.get_bad_mask()
    deltalns = []
    for time in oldtimes:
        quasar_data.remove_point(time)
        mags, times, bands, sigmas, bads = quasar_data.get_data()
        deltaln = qv.temp_ln(bestparams, mags, times, bands, sigmas, default, onofflist)-bestprob
        print deltaln
        deltalns.append(deltaln)
        print len(deltalns)
        quasar_data.bad = oldbad
    
