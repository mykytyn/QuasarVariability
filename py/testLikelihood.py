import numpy as np
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyfits
import mag_utils
import QuasarVariability


def graph_prior_and_posterior(data, prefix='quasar'):
    bands_dict = data.get_banddict()
    mags = data.get_mags()
    times = data.get_times()
    sigmas = data.get_sigmas()
    bands = data.get_bandnames()
    bandsnum = data.get_bands()
    bandlist = data.get_bandlist()

    tau = 200.

    dt = 10.
    initial_time = np.min(times)-25
    final_time = np.max(times)+25

    timegrid, bandsgrid = mag_utils.make_band_time_grid(initial_time,
                                                        final_time, dt,
                                                        bandlist)

    means, amps = mag_utils.grid_search_all_bands(mags, sigmas, bandsnum)
    print means, amps
    means = np.array(means)
    amps = np.array(amps)

    plt.clf()
    quasar = QuasarVariability.QuasarVariability(CovarianceFunction(amps, tau),
                                                 means)
    pmean = quasar.get_mean_vector(timegrid, bandsgrid)
    Vpp = quasar.get_variance_tensor(timegrid, bandsgrid)
    pmean = np.array(pmean).reshape(timegrid.shape)
    psig = np.sqrt(np.diag(np.array(Vpp)))
    mag_utils.make_prior_plots(quasar, timegrid, bandsgrid, pmean, psig,
                               means)
    plt.savefig("%s_prior.png" % prefix)

    plt.clf()
    pmean, Vpp = quasar.get_conditional_mean_and_variance(timegrid, bandsgrid,
                                                          mags, times,
                                                          bandsnum, sigmas)
    pmean = np.array(pmean).reshape(timegrid.shape)
    psig = np.sqrt(np.diag(np.array(Vpp)))
    mag_utils.make_posterior_plots(quasar, times, mags, bandsnum, sigmas,
                                   timegrid, bandsgrid, pmean, psig, means)
    plt.savefig("%s_posterior.png" % prefix)
