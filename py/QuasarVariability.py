import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import emcee
import utils

init_tau = 50.
wavelengths = [3543., 4770., 6231., 7625., 9134.] #THESE SHOULD NOT BE HARDCODED
base = 2

class RandomWalk:
    def __init__(self, pars, onofflist, wavelengths, base):
        """
        Inputs: pars, as a list containing:
        log(a_r): amplitude of the base band
        alpha: exponent for the amplitudes
        log(tau_r): scalar of time scale in the base band (in days)
        beta: exponent for the taus
        delta_r: time shift for the base band
        gamma: exponent for the time shifts
        and:
        wavelengths: wavelengths of the different bands
        base: which band is the base (index)

        Outputs: None
        Comments: Damped random walk model
        """
        self.wavelengths = wavelengths
        self.base = base
        self.onofflist = onofflist
        self.a_r = np.exp(pars[0])
        self.alpha = pars[1]
        assert not np.isnan(self.alpha)
        self.tau = np.exp(pars[2])
        self.delta_r = pars[3]
        self.gamma = pars[4]
        self.S = pars[5] #log?
        self.a = np.array([self._get_coef(x, self.a_r, self.alpha)
                           for x in range(len(self.wavelengths))])
        self.delta = np.array([self._get_coef(x, self.delta_r, self.gamma)
                               for x in range(len(self.wavelengths))])
        self.par_list = [self.a_r, self.alpha, self.tau,
                         self.delta_r, self.gamma, self.S]

    def _get_coef(self, band, coef, exponent):
        coef = coef * ((self.wavelengths[band] / self.wavelengths[self.base])
                       ** exponent)
        return coef

    def _get_kernal_matrix(self, t1, b1, t2, b2):
        return np.array(self.a[b1[:, None]] * self.a[b2[None, :]] *
                        np.exp(-1 * (np.abs(t1[:, None] - t2[None, :] +
                                            self.delta[b1[:, None]] -
                                            self.delta[b2[None, :]])) /
                        self.tau))

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
        tt = self._get_kernal_matrix(times, bands, times, bands)
        tt = tt + np.identity(tt.shape[0])*(self.S**2)
        if sigmas is None:
            return tt
        assert len(sigmas) == len(times)
        return tt + np.diag(sigmas ** 2)

    def set_pars(self, pars):
        counter = 0
        if self.onofflist[0]:
            self.a_r = np.exp(pars[counter])
            counter += 1
        if self.onofflist[1]:
            self.alpha = pars[counter]
            counter += 1
            assert not np.isnan(self.alpha)
        if self.onofflist[2]:
            self.tau = np.exp(pars[counter])
            counter += 1
        if self.onofflist[3]:
            self.delta_r = pars[counter]
            counter += 1
        if self.onofflist[4]:
            self.gamma = pars[counter]
            counter += 1
        if self.onofflist[5]:
            self.S = pars[counter] #log?
            counter +=1
        self.a = np.array([self._get_coef(x, self.a_r, self.alpha)
                           for x in range(len(self.wavelengths))])
        self.delta = np.array([self._get_coef(x, self.delta_r, self.gamma)
                               for x in range(len(self.wavelengths))])

    def get_pars(self):
        return self.a_r, self.alpha, self.tau, self.delta_r, self.gamma, self.S

    def get_packed_pars(self):
        packs = []
        for i, par, status in zip(range(len(self.par_list)),
                                  self.par_list, self.onofflist):
            if status:
                if i == 0 or i == 2:
                    packs.append(np.log(par))
                else:
                    packs.append(par)
        return packs

    def top_hat_prior(self, par, mean, var, lower, upper):
        if lower < par < upper:
            return utils.ln_1d_gauss(par, mean, var)
        else:
            return -np.inf

    def get_priors(self):
        prior = 0
        a_r, alpha, tau, delta_r, gamma, S = self.get_pars()
        if self.onofflist[0]:
            prior += utils.ln_1d_gauss(np.log(a_r), -1., 1.)
        if self.onofflist[1]:
            prior += utils.ln_1d_gauss(alpha, -1., .25)
        if self.onofflist[2]:
            prior += utils.ln_1d_gauss(np.log(tau), 6., 2.)
            #prior += self.top_hat_prior(np.log(tau), 5., 2., 3., 100.)
        if self.onofflist[3]:
            if delta_r<0:
                return -np.inf
            prior += utils.ln_1d_gauss(delta_r, 0., 1.)
        if self.onofflist[4]:
            prior += utils.ln_1d_gauss(gamma, -1., .25)  # irrelevant
        if self.onofflist[5]:
            if S<0:
                return -np.inf
            prior += 0 #CHANGE
        return prior

    def get_labels(self):
        labels = ['mean u', 'mean g', 'mean r', 'mean i', 'mean z']
        for status, par in zip(self.onofflist, ['ln a_r', 'alpha',
                                                'ln tau', 'delta_r', 'gamma','S']):
            if status:
                labels.append(par)
        return labels


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
        return np.array([self.get_mean()[bands]]).T

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
        Inputs: x is the variable (as a numpy 2d array)[column vector]
        mu is the mean (as a numpy 2darray)[column vector]
        V is the variance matrix
        Output: Natural log of the gaussian evalulated at x
        NOTE: there must be a way to compute determinant/inverse faster
        """
        assert len(x) == len(mu) == V.shape[0] == V.shape[1]
        dx = x - mu
        try:
            L, lower = linalg.cho_factor(V, lower=True)
            alpha = linalg.cho_solve((L.T, False),dx)
            #assert np.all(np.abs(alpha-np.dot(linalg.inv(V),dx))<10**-8)
            #assert np.all(np.abs(np.dot(L,L.T) - V) < 10**-8)
            logdet = 2*np.sum(np.log(np.diagonal(L)))
            #assert logdet - np.linalg.slogdet(V)[1] < 10**-8
            return -.5 * logdet + -0.5 * np.dot(dx.T, alpha)
        except:
            return -np.inf

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
            np.array([mags]).T,
            self.get_mean_vector(times, bands),
            self.get_variance_tensor(times, bands, sigmas))

    def ln_prior(self):
        """
        Priors still in progress!!!
        Use with caution!!!
        """
        prior = 0
        for i in self.get_mean():
            prior += utils.ln_1d_gauss(i, 20., 10.)
        prior += self.get_covar().get_priors()
        return prior

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
        Vpo = self.covar._get_kernal_matrix(ptimes, pbands, times, bands)
        Voo = self.covar.get_variance_tensor(times, bands, sigmas)
        VooI = np.linalg.inv(Voo)
        Vpp = self.covar.get_variance_tensor(ptimes, pbands)
        pmean = np.dot(np.dot(Vpo, VooI),
                       (np.array([mags]).T -
                        self.get_mean_vector(times, bands)))
        pmean = pmean + self.get_mean_vector(ptimes, pbands)
        Vpp = Vpp - np.dot(np.dot(Vpo, VooI), Vpo.T)
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
        prior = self.ln_prior()
        if np.isinf(prior):
            return -np.inf
        likelihood = self.ln_likelihood(mags, times, bands, sigmas)
        if np.isinf(likelihood):
            return -np.inf
        else:
            return likelihood[0, 0] + prior

    def get_labels(self):
        return self.get_covar().get_labels()


def temp_ln(pars, mags, times, bands, sigmas, default_pars,
            onofflist):
    tempObj = QuasarVariability(RandomWalk(default_pars[5:],
                                           onofflist, wavelengths,
                                           base),
                                default_pars[0:5])
    return tempObj.mc_ln_prob(pars, mags, times, bands, sigmas)
