import cPickle
import numpy as np
from scipy.optimize import leastsq
from scipy.misc import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils

def retrieve_tau_data(objids):
    """
    so as to save time
    since unpickling is slow
    lets load all data
    """

    alltaus = []
    for objid in objids:
        g = open("{}.pickle".format(objid))
        quasar, quasar_data, flatchain, lprobability, labels = cPickle.load(g)
        alltaus.append(np.exp(flatchain[:, 7]))
        g.close()
    return alltaus


def importance_sample_one(objid, taus, tau_mean, tau_variance):
    """
    takes in one id
    and also mean and sigma
    gives importance samp
    """

    return logsumexp(utils.ln_1d_gauss(taus, tau_mean, tau_variance))


def importance_sample_all(targlist, taus, tau_mean, tau_variance):
    """
    takes an id list
    and also mean and sigma
    returns total ln prob
    """

    total = 0
    for targ, tau in zip(targlist, taus):
        total += importance_sample_one(targ, tau, tau_mean, tau_variance)
    return total


def ln_prior(tau_mean, tau_variance):
    """
    this is only temp
    waiting for more information
    should cut small sigma
    """

    return 0


def ln_prob(targets, taus, tau_mean, tau_variance):
    """
    this takes a targ list
    and hyperparameters
    and gives the ln prob
    """

    prior = ln_prior(tau_mean, tau_variance)
    if np.isinf(prior):
        return -np.inf
    return prior + importance_sample_all(targets, taus, tau_mean, tau_variance)


def make_large_triangle_plot():
    """
    still working on it
    everything is hardcoded
    might be removed soon
    """

    f = open('newtargetlist.txt', 'r')
    hugeflatchains = []
    hugelnprobabilities = []
    for numobj, obj in enumerate(f):
        prefix = obj
        obj = int(obj)
        g = open("{}.pickle".format(obj))
        quasar, quasar_data, flatchain, lnprobability, labels = cPickle.load(g)
        g.close()
        hugeflatchains.append(flatchain)
        hugelnprobabilities.extend(lnprobability)
        print flatchain.shape, lnprobability.shape, len(labels)
    hugeflatchain = np.concatenate(hugeflatchains)
    print hugeflatchain.shape
    hugelnprobability = np.hstack(hugelnprobabilities)
    print hugelnprobability.shape
    utils.make_triangle_plot(hugelnprobability,
                             hugeflatchain, labels).savefig('hugetriangle.png')

def main():
    f = open('newtargetlist.txt', 'r')
    targets = [int(x) for x in f]
    f.close()
    taus = retrieve_tau_data(targets)
    tau_variances = np.arange(1, 300, 1.)
    ln_probs = []
    for x in tau_variances:
        ln_probs.append(ln_prob(targets, taus, 200, x))
    plt.plot(tau_variances, ln_probs, '+')
    plt.xlabel('Tau Variance')
    plt.ylabel('Ln Prob')
    plt.savefig('newsample2.png')



if __name__ == '__main__':
    main()

