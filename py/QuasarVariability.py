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
wavelengths = [3543.,4770.,6231.,7625.,9134.]
base = 2


class RandomWalk:
    def __init__(self, a, tau, fixedTau=False):
        """
        Inputs: a: amplitudes of each band
                tau: scalar of time scale (in days)
        Outputs: None
        Comments: Damped random walk model
        """

        self.a = a
        self.tau = tau
        self.fixedTau = fixedTau


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

    def set_pars(self, pars):
        self.a = pars[0:5]
        if not fixedTau:
            self.tau = pars[6]

    def get_pars(self):
        if not self.fixedTau:
            return self.a, self.tau
        return self.a

    def get_packed_pars(self):
        if not self.fixedTau:
            return np.log(self.a), np.log(self.tau)
        return np.log(self.a)

    def get_priors(self):
        """
        Priors still in progress!!!
        Use with caution!!!
        """
        prior = 0.
        if self.fixedTau:
            amps = self.get_pars()
        else:
            amps,tau = self.get_pars()
            prior += utils.ln_1d_gauss(tau, 5., 2.)
        for a in amps:
            prior += utils.ln_1d_gauss(a, -1. ,1)
        return prior

    def get_labels(self):
        if self.fixedTau:
            return ['mean u', 'mean g', 'mean r', 'mean i',
                    'mean z', 'ln a_u', 'ln a_g', 'ln a_r', 
                    'ln a_i', 'ln a_z','ln_tau']
        else:
            return ['mean u', 'mean g', 'mean r', 'mean i',
                    'mean z', 'ln a_u', 'ln a_g', 'ln a_r', 
                    'ln a_i', 'ln a_z']
            



class RandomWalkAlpha:
    def __init__(self, a_r, alpha, tau, wavelengths, base, fixedTau = False):
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
        self.a = np.array([self._get_coef(x) for x in range(len(self.wavelengths))])
        self.fixedTau = fixedTau

    def _get_coef(self, band):
        coef =  self.a_r * ((self.wavelengths[band] / self.wavelengths[self.base]) ** self.alpha)
        return coef

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

    def set_pars(self, pars):
        self.a_r = np.exp(pars[0])
        self.alpha = pars[1]
        assert not np.isnan(self.alpha)
        if not self.fixedTau:
            self.tau = np.exp(pars[2])
        self.a = np.array([self._get_coef(x) for x in range(len(self.wavelengths))])

    def get_pars(self):
        if not self.fixedTau:
            return self.a_r, self.alpha, self.tau
        return self.a_r, self.alpha

    def get_packed_pars(self):
        if not self.fixedTau:
            return np.log(self.a_r), self.alpha, np.log(self.tau)
        else:
            return np.log(self.a_r), self.alpha

    def get_priors(self):
        prior = 0
        if not self.fixedTau:
            a_r, alpha, tau = self.get_pars()
            prior += utils.ln_1d_gauss(tau, 5., 2.)
        else:
            a_r, alpha = self.get_pars()
        prior += utils.ln_1d_gauss(a_r, -1. ,1)
        prior += utils.ln_1d_gauss(alpha, -1., .25)
        return prior

    def get_labels(self):
        if self.fixedTau:
            return ['mean u', 'mean g', 'mean r', 'mean i', 'mean z',
                    'ln a_r', 'alpha', 'ln_tau']
        else:
            return ['mean u', 'mean g', 'mean r', 'mean i', 'mean z',
                    'ln a_r', 'alpha']


class newRandomWalk:
    def __init__(self, a_r, alpha, tau_r, beta, delta_r, gamma, wavelengths, base):
        """
        Inputs: a: amplitudes of each band
                tau: scalar of time scale (in days)
        Outputs: None
        Comments: Damped random walk model
        """

        self.a_r = a_r
        self.alpha = alpha
        self.tau_r = tau_r
        self.beta = beta
        self.delta_r = delta_r
        self.gamma = gamma
        self.wavelengths = wavelengths
        self.base = base
        self.a = np.array([self._get_coef(x,a_r,alpha) for x in range(len(self.wavelengths))])
        self.tau = np.array([self._get_coef(x,tau_r,beta) for x in range(len(self.wavelengths))])
        self.delta = np.array([self._get_coef(x,delta_r,gamma) for x in range(len(self.wavelengths))])


    def _get_coef(self, band, coef, exponent):
        coef =  coef * ((self.wavelengths[band] / self.wavelengths[self.base]) ** exponent)
        return coef

    def _get_cross_term_matrix(self, t1, b1, t2, b2):
        return np.matrix(self.a[b1[:, None]] * self.a[b2[None, :]] *
                         np.exp(-1 * (np.abs(t1[:,None] - t2[None, :] + self.delta[b1[:, None]] - self.delta[b2[None,:]])) / np.sqrt(self.tau[b1[:, None]] * self.tau[b2[None,:]])))

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

    def set_pars(self, pars):
        self.a_r = np.exp(pars[0])
        self.alpha = pars[1]
        assert not np.isnan(self.alpha)
        self.tau_r = np.exp(pars[2])
        self.beta = pars[3]
        self.delta_r = pars[4]
        self.gamma = pars[5]
        self.a = np.array([self._get_coef(x,self.a_r,self.alpha) for x in range(len(self.wavelengths))])
        self.tau = np.array([self._get_coef(x,self.tau_r,self.beta) for x in range(len(self.wavelengths))])
        self.delta = np.array([self._get_coef(x,self.delta_r,self.gamma) for x in range(len(self.wavelengths))])

    def get_pars(self):
        return self.a_r, self.alpha, self.tau_r, self.beta, self.delta_r, self.gamma

    def get_packed_pars(self):
        return np.log(self.a_r), self.alpha, np.log(self.tau_r), self.beta, self.delta_r, self.gamma

    def get_priors(self):
        #TODO: FIX PRIORS OK
        prior = 0
        a_r, alpha, tau_r, beta, delta_r, gamma = self.get_pars()
        prior += utils.ln_1d_gauss(tau_r, 5., 2.)
        prior += utils.ln_1d_gauss(beta, -1., .25)
        prior += utils.ln_1d_gauss(delta_r, 0., 1.)
        prior += utils.ln_1d_gauss(gamma, -1., .25)
        prior += utils.ln_1d_gauss(a_r, -1. ,1.)
        prior += utils.ln_1d_gauss(alpha, -1., .25)
        return prior

    def get_labels(self):
        return ['mean u', 'mean g', 'mean r', 'mean i', 'mean z',
                'ln a_r', 'alpha', 'ln tau_r', 'beta', 'ln delta_r',
                'gamma', 'ln_prob']


class QuasarVariability:
    def __init__(self, covar, mean):
        """
        Inputs: covar: a covariance function object
                mean: a five-vector of the mean in each band
        Outputs:
        Comments:
        """
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
        try:
            return -0.5 * np.linalg.slogdet(V)[1] + \
                -0.5 * dx.getT() * V.getI() * dx
        except:
            print self.get_covar().get_pars()
            raise
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

    def pack_pars(self):
        temp = []
        temp.extend(self.get_mean())
        temp.extend(self.get_covar().get_packed_pars())
        return temp

    def unpack_pars(self, pars):
        self.set_mean(pars[0:5])
        self.get_covar().set_pars(pars[5:])

    def mc_ln_prob(self, pars, mags, times, bands, sigmas):
        self.unpack_pars(pars)
        return (self.ln_likelihood(mags, times, bands, sigmas)[0, 0]
                + self.ln_prior())

    def get_labels(self):
        return self.get_covar().get_labels()


    


def temp_ln(pars, mags, times, bands, sigmas, alpha=False, noTau=False, newCovar=False):
    if newCovar:
        tempObj = QuasarVariability(newRandomWalk(-1.,-1.,init_tau,-1.,0.,-1.,wavelengths,base), pars[0:5])
    elif alpha:
        tempObj = QuasarVariability(RandomWalkAlpha(
                np.exp(pars[5]), np.exp(pars[6]), init_tau,
                wavelengths, base, noTau),
                                    pars[0:5])
    else:
        tempObj = QuasarVariability(RandomWalk(
                np.exp(pars[5:10]), init_tau, noTau), pars[0:5])
    return tempObj.mc_ln_prob(pars, mags, times, bands, sigmas)




def run_mcmc(data, prefix, num_steps, initialp0, noTau=False, alpha=False, newCovar=False):
    mags = data.get_mags()
    sigmas = data.get_sigmas()
    bands = data.get_bands()
    times = data.get_times()
    bandnames = data.get_bandlist()

    dt = 30.
    initialtime = np.min(times) - 50
    finaltime = np.max(times) + 50
    timegrid, bandgrid = utils.make_band_time_grid(initialtime, finaltime,
                                                       dt, bandnames)
    if newCovar:
        init_means = initialp0[0:5]
        init_a_r = initialp0[5]
        init_alpha = initialp0[6]
        init_tau_r = initialp0[7]
        init_beta = initialp0[8]
        init_delta_r = initialp0[9]
        init_gamma = initialp0[10]
    if alpha:
        qv = QuasarVariability(RandomWalkAlpha(amps[2],-1.,init_tau, wavelengths, base, noTau), means)
    elif newCovar:
        qv = QuasarVariability(newRandomWalk(init_a_r,init_alpha,
                                             init_tau_r,init_beta,init_delta_r,init_gamma,wavelengths,base), init_means)
    else:
        qv = QuasarVariability(RandomWalk(amps, init_tau, noTau), means)

    if alpha:
        ndim = 8 - noTau
    elif newCovar:
        ndim = 11
    else:
        ndim = 11 - noTau

    nwalkers = 32
    nthreads = 10

    p0 = qv.pack_pars()
    print p0

    initial = []
    for i in range(nwalkers):  # could probably be improved -mykytyn
        pp = p0 + 0.0001 * np.random.normal(size=len(p0))  # Magic number
        initial.append(pp)
    arguments = [mags,times, bands, sigmas, alpha, noTau, newCovar]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, temp_ln,
                                    args=arguments, threads=nthreads)

    pos, prob, state = sampler.run_mcmc(initial, 1)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, num_steps)
    labels = qv.get_labels()

    highln = np.argmax(sampler.lnprobability)
    best = sampler.flatchain[highln]
    bestln = sampler.lnprobability.flatten()[highln]
    best[5] = np.exp(best[5])
    best[7] = np.exp(best[7])
    trichain = np.column_stack((sampler.flatchain, sampler.lnprobability.flatten()))

    figure = triangle.corner(trichain, labels=labels)
    figure.savefig('%s-triangle.png' % (prefix))
    plt.clf()
    chain = sampler.chain
    sample = sampler.flatchain[-1, :]

    #quasarsamp = QuasarVariability(RandomWalk(sample[0:5],init_tau),sample[5:10])
    #pmean, Vpp = quasarsamp.get_conditional_mean_and_variance(timegrid,bandgrid,mags,times,bands, sigmas)
    #pmean = np.array(pmean).reshape(timegrid.shape)
    #psig = np.sqrt(np.diag(np.array(Vpp)))
    #utils.make_posterior_plots(quasarsamp, times, mags, bands, sigmas, timegrid, bandgrid, pmean, psig, means, bandnames)
    #plt.savefig('{}-{}-posterior.png'.format(prefix,objid))
    
    for j,par in enumerate(labels):
        plt.clf()
        if par == "ln_prob":
            for i in range(nwalkers):
                plt.plot(sampler.lnprobability[i, :])
        else:
            for i in range(nwalkers):
                plt.plot(chain[i, :, j])
        plt.xlabel('Step Number')
        plt.ylabel('{}'.format(par))
        plt.savefig('%s-walker-%s.png' % (prefix, par))

    return best, bestln


def main():
    #obj = 588015509825912905  # lot of data
    #obj = 587730845812064296  # less data

    for obj in open('targets.txt'):
        run_mcmc(int(obj), 'targets')

if __name__ == '__main__':
    main()
