import numpy as np

class QuasarVariability:
    def __init__(self, a, tau, mean):
        """
        inputs:
        outputs:
        comments:
          - For now, a, tau, mean are all scalars.
        """
        # DM put in asserts that everything is scalar.
        self.a = a
        self.tau = tau
        self.mean = mean

    def get_mean_vector(self, times):
        """
        inputs:
        outputs:
        comments:
        """
        return np.matrix(np.zeros_like(times) + self.mean).transpose()

    def _get_cross_term_matrix(self, t1, t2):
        return np.matrix(self.a ** 2 * np.exp(-1. / self.tau * np.abs(t1[:, None] -
                                                                       t2[None, :])))

    def get_variance_tensor(self, times, sigmas=None):
        """
        inputs:
        outputs:
        comments:
        """
        tt = self._get_cross_term_matrix(times, times)
        if sigmas is None:
            return tt
        assert len(sigmas) == len(times)
        return tt + np.diag(sigmas ** 2)

    def get_prior_sample(self, times):
        """
        inputs:
        outputs:
        comments:
        """
        return np.random.multivariate_normal(np.array(self.get_mean_vector(times)).reshape(times.shape),self.get_variance_tensor(times),1)

    def _ln_multivariate_gaussian(self, x, mu, V):
        """
        Compute log of Nd Gaussian
        Inputs: x is the variable (as a numpy matrix)[column vector]
        mu is the mean (as a numpy matrix)[column vector]
        V is the variance matrix
        Output: Natural log of the gaussian evalulated at x
        NOTE: there must be a way to compute determinant/inverse faster
        """

        A=1/np.sqrt((2*np.pi)**len(x))
        assert len(x)==len(mu)==V.shape[0]==V.shape[1]
        dx = x-mu
        return (np.log(A) - 0.5 * np.linalg.slogdet(V)[1] - 
                0.5 * dx.getT() * V.getI() * dx)


    def ln_likelihood(self, mags, times, sigmas):
        """
        inputs:
        outputs:
        comments:
        """
        return self._ln_multivariate_gaussian(np.matrix(mags).getT(),
                                              self.get_mean_vector(times),
                                              self.get_variance_tensor(times, sigmas))

    def get_conditional_mean_and_variance(self, ptimes, mags, times, sigmas):
        """
        inputs:
        outputs:
        comments:
        """
        Vpo = self._get_cross_term_matrix(ptimes, times)
        Voo = self.get_variance_tensor(times, sigmas)
        VooI = Voo.getI()
        Vpp = self.get_variance_tensor(ptimes)
        pmean = Vpo * VooI * (np.matrix(mags).getT() - self.get_mean_vector(times)) \
            + self.get_mean_vector(ptimes)
        Vpp = Vpp - Vpo * VooI * Vpo.getT()
        return pmean,Vpp
        
    def get_conditional_sample(self, ptimes, mags, times, sigmas):
        pmean,Vpp = self.get_conditional_mean_and_variance(ptimes,mags,times,sigmas)
        return np.random.multivariate_normal(np.array(pmean).reshape(ptimes.shape), Vpp, 1)
