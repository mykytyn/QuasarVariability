import numpy as np
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyfits
import emcee
import utils
import triangle
import stripe82

init_tau = 50.

class RandomWalk:
    def __init__(self, a, tau):
        """
        Inputs: a: amplitudes of each band
                tau: scalar of time scale (in days)
        Outputs: None
        Comments: Damped random walk model
        """

        self.a = a
        self.tau = tau


    def _get_cross_term_matrix(self, t1, b1, t2, b2):
        return np.matrix(self.a[b1[:, None]] * self.a[b2[None, :]] *
                         np.exp(-1. / self.tau * np.abs(t1[:, None] -
                                                        t2[None, :])))

    def get_variance_tensor(self, times, bands, sigmas=None):
        """
        Inputs: times: an array of times (in days)
                bands: a vector associating each point with a band
                       (using 0=u, etc.)
                sigmas: an array of variances
        Outputs:
                the variance tensor
        Comments:
        """
        tt = self._get_cross_term_matrix(times, bands, times, bands)
        if sigmas is None:
            return tt
        assert len(sigmas) == len(times)
        return tt + np.diag(sigmas ** 2)

    def set_pars(self, a, tau):
        self.a = a
        self.tau = tau

    def get_pars(self):
        return self.a, self.tau

    def get_priors(self):
        """
        Priors still in progress!!!
        Use with caution!!!
        """
        amps,tau = self.get_pars()
        prior = 0.
        for a in amps:
            prior += utils.ln_1d_gauss(a, -1. ,1)
        prior += utils.ln_1d_gauss(tau, 5., 2.)
        return prior


class RandomWalkAlpha:
    def __init__(self, a_r, alpha, tau, wavelengths, base):
        """
        Inputs: a: amplitudes of each band
                tau: scalar of time scale (in days)
        Outputs: None
        Comments: Damped random walk model
        """

        self.a_r = a_r
        self.alpha = alpha
        self.tau = tau
        self.wavelengths = wavelengths
        self.base = base
        self.a = [self._get_coef(x) for x in range(len(self.wavelengths))]

    def _get_coef(self, band):
        return a_r * ((self.wavelengths[band] / self.wavelengths[self.base]) ** self.alpha)

    def _get_cross_term_matrix(self, t1, b1, t2, b2):
        return np.matrix(self.a[b1[:, None]] * self.a[b2[None, :]] *
                         np.exp(-1. / self.tau * np.abs(t1[:, None] -
                                                        t2[None, :])))

    def get_variance_tensor(self, times, bands, sigmas=None):
        """
        Inputs: times: an array of times (in days)
                bands: a vector associating each point with a band
                       (using 0=u, etc.)
                sigmas: an array of variances
        Outputs:
                the variance tensor
        Comments:
        """
        tt = self._get_cross_term_matrix(times, bands, times, bands)
        if sigmas is None:
            return tt
        assert len(sigmas) == len(times)
        return tt + np.diag(sigmas ** 2)

    def set_pars(self, a_r, alpha, tau, wavelengths, base):
        self.a_r = a_r
        self.alpha = alpha
        self.tau = tau
        self.wavelengths = wavelengths
        self.base = base
        self.a = [self._get_coef(x) for x in range(len(self.wavelengths))]


    def get_pars(self):
        return self.a_r, self.alpha, self.tau

    def get_priors(self):
        a_r, alpha, tau = self.get_pars()
        prior = utils.ln_1d_gauss(tau, 5., 2.)
        prior += utils.ln_1d_gauss(a_r, -1. ,1)
        prior += utils.ln_1d_gauss(alpha, -1., .25)
        return prior

class QuasarVariability:
    def __init__(self, covar, mean):
        """
        Inputs: covar: a covariance function object
                mean: a five-vector of the mean in each band
        Outputs:
        Comments:
        """
        assert len(mean) == 5
        self.set_covar(covar)
        self.set_mean(mean)

    def get_covar(self):
        return self.covar

    def set_covar(self, c):
        self.covar = c

    def get_mean(self):
        return self.mean

    def set_mean(self, m):
        self.mean = m

    def get_variance_tensor(self, times, bands, sigmas=None):
        """
        Inputs: times: an array of times (in days)
                bands: a vector associating each point with a band
                       (using 0=u, etc.)
                sigmas: an array of variances
        Outputs:
                the variance tensor
        Comments:
        """
        return self.covar.get_variance_tensor(times, bands, sigmas)

    def get_mean_vector(self, times, bands):
        """
        Comments: This propogates the means at each band at each time
        """
        return np.matrix(self.get_mean()[bands]).transpose()

    def get_prior_sample(self, times, bands):
        """
        Inputs: times: an array of times we want to sample at
                bands: an array of bands associated with each time
        Outputs:
                a draw from the prior
        Comments:
        """
        return np.random.multivariate_normal(
            np.array(self.get_mean_vector(times, bands)).reshape(times.shape),
            self.get_variance_tensor(times, bands), 1)

    def _ln_multivariate_gaussian(self, x, mu, V):
        """
        Compute log of Nd Gaussian
        Inputs: x is the variable (as a numpy matrix)[column vector]
        mu is the mean (as a numpy matrix)[column vector]
        V is the variance matrix
        Output: Natural log of the gaussian evalulated at x
        NOTE: there must be a way to compute determinant/inverse faster
        """
        assert len(x) == len(mu) == V.shape[0] == V.shape[1]
        dx = x - mu
        return -0.5 * np.linalg.slogdet(V)[1] + \
               -0.5 * dx.getT() * V.getI() * dx

    def ln_likelihood(self, mags, times, bands, sigmas):
        """
        Inputs: mags: an array of mags
                times: an array of times
                bands: an array of bands associated with each time
                sigmas: the variances
        Outputs:
                The ln of the likelihood function
        """
        return self._ln_multivariate_gaussian(
            np.matrix(mags).getT(),
            self.get_mean_vector(times, bands),
            self.get_variance_tensor(times, bands, sigmas))

    def ln_prior(self):
        """
        Priors still in progress!!!
        Use with caution!!!
        """
        return self.get_covar().get_priors()
            

    def get_conditional_mean_and_variance(self, ptimes, pbands,
                                          mags, times, bands, sigmas):
        """
        Inputs: ptimes: array of times to test at
                pbands: array of bands matching the ptimes
                mags: measured mags
                times: measured times
                bands: measured bands
                sigmas: measured variances
        Outputs: pmean: mean at ptimes
                 Vpp: covariance matrix
        """
        Vpo = self.covar._get_cross_term_matrix(ptimes, pbands, times, bands)
        Voo = self.covar.get_variance_tensor(times, bands, sigmas)
        VooI = Voo.getI()
        Vpp = self.covar.get_variance_tensor(ptimes, pbands)
        pmean = (Vpo * VooI * (np.matrix(mags).getT() -
                               self.get_mean_vector(times, bands))
                 + self.get_mean_vector(ptimes, pbands))
        Vpp = Vpp - Vpo * VooI * Vpo.getT()
        return pmean, Vpp

    def get_conditional_sample(self, ptimes, pbands, mags, times, bands,
                               sigmas):
        """
        Inputs: ptimes: array of times to sample at
                pbands: array of bands matching the ptimes
                mags: measured mags
                times: measured times
                bands: measured bands
                sigmas: measured variances
        Outputs: A draw from the multivariate gaussian at ptimes
        Comments:
        """
        pmean, Vpp = self.get_conditional_mean_and_variance(ptimes, pbands,
                                                            mags, times,
                                                            bands, sigmas)
        return np.random.multivariate_normal(
            np.array(pmean).reshape(ptimes.shape), Vpp, 1)

    def pack_pars(self, noTau=False):
        temp = []
        temp.extend(self.get_mean())
        pars = self.get_covar().get_pars()
        if len(pars)==2:
            a, tau = pars
            temp.extend(np.log(a))
        else:
            a_r, alpha, tau = pars
            temp.append(np.log(a_r))
            temp.append(np.log(alpha))
        if not noTau:
            temp.append(np.log(tau))
        return temp


    def unpack_pars(self, pars):
        self.set_mean(pars[0:5])
        if len(pars)==11:
            self.get_covar().set_pars(np.exp(pars[5:10]), np.exp(pars[10]))
        elif len(pars)==10:
            self.get_covar().set_pars(np.exp(pars[5:10]), self.get_covar().get_pars()[1])
        elif len(pars)==8:
            self.get_covar().set_pars(np.exp(pars[5]), np.exp(pars[6]), np.exp(pars[7]))
        else:
            self.get_covar().set_pars(np.exp(pars[5]), np.exp(pars[6]))

    def mc_ln_prob(self, pars, mags, times, bands, sigmas):
        self.unpack_pars(pars)
        return (self.ln_likelihood(mags, times, bands, sigmas)[0, 0]
                + self.ln_prior())


def temp_ln(pars, mags, times, bands, sigmas):
    tempObj = QuasarVariability(RandomWalk(pars[0:5], init_tau),
                                pars[5:10])
    return tempObj.mc_ln_prob(pars, mags, times, bands, sigmas)


def run_mcmc(objid, prefix):
    data = stripe82.Stripe82(objid)
    mags = data.get_mags()
    print len(mags)
    sigmas = data.get_sigmas()
    bands = data.get_bands()
    times = data.get_times()

    bandnames = data.get_bandlist()

    dt = 30.
    initialtime = np.min(times) - 50
    finaltime = np.max(times) + 50

    timegrid, bandgrid = utils.make_band_time_grid(initialtime, finaltime,
                                                       dt, bandnames)

    means, amps = utils.grid_search_all_bands(mags, sigmas, bands)


    qv = QuasarVariability(RandomWalk(amps, init_tau), means)

    ndim = 10
    nwalkers = 26
    nthreads = 10
    p0 = []
    p0.extend(means)
    p0.extend(amps)
    #p0.append(init_tau)
    p0 = np.array(p0)

    initial = []
    p0 = qv.pack_pars(noTau=True)
    for i in range(nwalkers):  # could probably be improved -mykytyn
        pp = p0 + 0.0001 * np.random.normal(size=len(p0))  # Magic number
        initial.append(pp)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, temp_ln,
                                    args=[mags, times, bands, sigmas],
                                    threads=nthreads)

    pos, prob, state = sampler.run_mcmc(initial, 500)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, 500)
    if len(p0)==11:
        labels = ['ln a_u', 'ln a_g', 'ln a_r', 'ln a_i', 'ln a_z',
                  'mean u', 'mean g', 'mean r', 'mean i', 'mean z','ln_tau']

    else:
        labels = ['ln a_u', 'ln a_g', 'ln a_r', 'ln a_i', 'ln a_z',
                  'mean u', 'mean g', 'mean r', 'mean i', 'mean z']
        
    figure = triangle.corner(sampler.flatchain, labels=labels)
    figure.savefig('%s-%d-triangle.png' % (prefix,objid))
    plt.clf()
    chain = sampler.chain
    print chain.shape
    print chain[0, :, 2].shape
    sample = sampler.flatchain[-1, :]
    quasarsamp = QuasarVariability(RandomWalk(sample[0:5],init_tau),sample[5:10])
    pmean, Vpp = quasarsamp.get_conditional_mean_and_variance(timegrid,bandgrid,mags,times,bands, sigmas)
    pmean = np.array(pmean).reshape(timegrid.shape)
    psig = np.sqrt(np.diag(np.array(Vpp)))
    utils.make_posterior_plots(quasarsamp, times, mags, bands, sigmas, timegrid, bandgrid, pmean, psig, means, bandnames)
    plt.savefig('{}-{}-posterior.png'.format(prefix,objid))
    
    for j in range(ndim):
        plt.clf()
        for i in range(nwalkers):
            plt.plot(chain[i, :, j])
        plt.savefig('%s-%d-walker-dim%d.png' % (prefix,objid, j))


def main():
    #obj = 588015509825912905  # lot of data
    #obj = 587730845812064296  # less data

    for obj in open('targets.txt'):
        run_mcmc(int(obj), 'targets')

if __name__ == '__main__':
    main()
