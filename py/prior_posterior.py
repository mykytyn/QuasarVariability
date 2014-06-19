import numpy as np
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyfits
import utils
from QuasarVariability import *
import stripe82

def graph_prior_and_posterior(data, pars, onofflist, default, prefix='quasar'):
    bands_dict = data.get_banddict()
    mags = data.get_mags()
    times = data.get_times()
    sigmas = data.get_sigmas()
    bands = data.get_bandnames()
    bandsnum = data.get_bands()
    bandlist = data.get_bandlist()
    medians = []
    for i in range(5):
        mask = [bandsnum == i]
        medians.append(np.median(mags[mask]))

    print medians
    dt = 25.
    initial_time = np.min(times)-25
    final_time = np.max(times)+25

    timegrid, bandsgrid = utils.make_band_time_grid(initial_time,
                                                        final_time, dt,
                                                        bandlist)
    print "best pars ", pars
    quasar = QuasarVariability(newRandomWalk(default[5:], onofflist, wavelengths, base), default[0:5])
    quasar.unpack_pars(pars)
    plt.clf()
    pmean = quasar.get_mean_vector(timegrid, bandsgrid)
    Vpp = quasar.get_variance_tensor(timegrid, bandsgrid)
    pmean = np.array(pmean).reshape(timegrid.shape)
    psig = np.sqrt(np.diag(np.array(Vpp)))
    print pmean, Vpp, pmean, psig
    utils.make_prior_plots(quasar, timegrid, bandsgrid, pmean, psig, medians,
                               bandlist)
    plt.savefig("%s-prior.png" % prefix)

    plt.clf()
    pmean, Vpp = quasar.get_conditional_mean_and_variance(timegrid, bandsgrid,
                                                          mags, times,
                                                          bandsnum, sigmas)
    pmean = np.array(pmean).reshape(timegrid.shape)
    psig = np.sqrt(np.diag(np.array(Vpp)))
    utils.make_posterior_plots(quasar, times, mags, bandsnum, sigmas,
                                   timegrid, bandsgrid, pmean, psig, medians,bandlist)
    plt.savefig("%s-posterior.png" % prefix)


def main():
    objid = 587730845812064296
    data = stripe82.Stripe82(objid)
    graph_prior_and_posterior(data,prefix='obj-{}'.format(objid))


if __name__ == '__main__':
    main()
