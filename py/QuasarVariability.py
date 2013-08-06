import numpy as np


class CovarianceFunction:
    def __init__(self, a, tau):
        """
        Inputs: a: five-vector of amplitudes in each band
                tau: scalar of time scale (in days)
        Outputs: None
        Comments: Damped random walk model
        """

        assert len(a) == 5
        self.a = a
        self.tau = tau

    def _get_cross_term_matrix(self, t1, b1, t2, b2):
        return np.matrix(self.a[b1[:, None]] * self.a[b2[None, :]] *
                         np.exp(-1. / self.tau * np.abs(t1[:, None] - t2[None, :])))

    def get_variance_tensor(self, times, bands, sigmas=None):
        """
        Inputs: times: an array of times (in days)
                bands: a vector associating each point with a band (using 0=u, etc.)
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


class QuasarVariability:
    def __init__(self, covar, mean):
        """
        Inputs: covar: a covariance function object
                mean: a five-vector of the mean in each band
        Outputs:
        Comments:
        """
        assert len(mean) == 5
        self.covar = covar
        self.mean = mean

    def get_variance_tensor(self, times, bands, sigmas=None):
        return self.covar.get_variance_tensor(times, bands, sigmas)

    def get_mean_vector(self, times, bands):
        """
        Inputs: times: an array of times (in days)
                bands: a vector associating each point with a band (using 0=u, etc.)
                sigmas: an array of variances
        Outputs:
                the variance tensor
        Comments:
        """
        return np.matrix([self.mean[x] for x in bands]).transpose()

    def get_prior_sample(self, times, bands):
        """
        Inputs: times: an array of times we want to sample at
                bands: an array of bands associated with each time
        Outputs:
                a draw from the prior
        Comments:
        """
        return np.random.multivariate_normal(np.array(self.get_mean_vector(times, bands)).reshape(times.shape), self.get_variance_tensor(times, bands), 1)

    def _ln_multivariate_gaussian(self, x, mu, V):
        """
        Compute log of Nd Gaussian
        Inputs: x is the variable (as a numpy matrix)[column vector]
        mu is the mean (as a numpy matrix)[column vector]
        V is the variance matrix
        Output: Natural log of the gaussian evalulated at x
        NOTE: there must be a way to compute determinant/inverse faster
        """

        A = 1 / np.sqrt((2 * np.pi) ** len(x))
        assert len(x) == len(mu) == V.shape[0] == V.shape[1]
        dx = x - mu
        return (np.log(A) - 0.5 * np.linalg.slogdet(V)[1] -
                0.5 * dx.getT() * V.getI() * dx)

    def ln_likelihood(self, mags, times, bands, sigmas):
        """
        Inputs: mags: an array of mags
                times: an array of times
                bands: an array of bands associated with each time
                sigmas: the variances
        Outputs:
                The ln of the likelihood function
        Comments:
        """
        return self._ln_multivariate_gaussian(np.matrix(mags).getT(),
                                              self.get_mean_vector(times, bands),
                                              self.get_variance_tensor(times, bands, sigmas))

    def get_conditional_mean_and_variance(self, ptimes, pbands, mags, times, bands, sigmas):
        """
        Inputs: ptimes: array of times to test at
                pbands: array of bands matching the ptimes
                mags: measured mags
                times: measured times
                bands: measured bands
                sigmas: measured variances
        Outputs: pmean: mean at ptimes
                 Vpp: covariance matrix
        Comments:
        """
        Vpo = self.covar._get_cross_term_matrix(ptimes, pbands, times, bands)
        Voo = self.covar.get_variance_tensor(times, bands, sigmas)
        VooI = Voo.getI()
        Vpp = self.covar.get_variance_tensor(ptimes, pbands)
        pmean = Vpo * VooI * (np.matrix(mags).getT() - self.get_mean_vector(times, bands)) \
            + self.get_mean_vector(ptimes, pbands)
        Vpp = Vpp - Vpo * VooI * Vpo.getT()
        return pmean, Vpp

    def get_conditional_sample(self, ptimes, pbands, mags, times, bands, sigmas):
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
        pmean, Vpp = self.get_conditional_mean_and_variance(ptimes, pbands, mags, times,
                                                            bands, sigmas)
        return np.random.multivariate_normal(np.array(pmean).reshape(ptimes.shape), Vpp, 1)
