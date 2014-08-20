import cPickle
import numpy as np
from scipy.optimize import leastsq
from scipy.misc import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
import cProfile
import re
import emcee
from multiprocessing import Pool

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
        alltaus.append(flatchain[:, 7])
        g.close()
    return alltaus


def importance_sample_one(taus, tau_mean, tau_variance):
    """
    takes in one id
    and also mean and sigma
    gives importance samp
    """

    return logsumexp(utils.ln_1d_gauss(taus, tau_mean, tau_variance))


def importance_sample_all(taus, tau_mean, tau_variance):
    """
    takes an id list
    and also mean and sigma
    returns total ln prob
    """

    total = 0
    for tau in taus:
        total += importance_sample_one(tau, tau_mean, tau_variance)
    return total


def ln_prior(tau_mean, tau_variance):
    """
    this is only temp
    waiting for more information
    should cut small sigma
    """

    if tau_variance > 0:
        return 0
    return -np.inf

def ln_prob(pars, taus):
    """
    this takes a targ list
    and hyperparameters
    and gives the ln prob
    """
    
    tau_mean, tau_variance = pars
    prior = ln_prior(tau_mean, tau_variance)
    if np.isinf(prior):
        return -np.inf
    return prior + importance_sample_all(taus, tau_mean, tau_variance)


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
    prefix = 'first'
    targets = [int(x) for x in f]
    f.close()
    taus = retrieve_tau_data(targets)
    nwalkers = 8
    ndim = 2
    num_steps = 512
    pool = Pool(10)
    arguments = [taus,]
    labels = ['tau_mean', 'tau_variance', 'ln_prob']
    p0 = [5., 2.]
    initial = []
    for i in range(nwalkers):
        pp = p0 + .0001 * np.random.normal(size=len(p0))
        initial.append(pp)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=arguments, pool=pool)
    pos, prob, state = sampler.run_mcmc(initial, num_steps)
    print "triangle plot"
    utils.make_triangle_plot(sampler.lnprobability, sampler.flatchain, labels, temp=True).savefig('{}-triangle.png'.format(prefix))
    print "walker plots"
    walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
    for par,plot in zip(labels,walker_plots):
        plot.savefig('{}-walker-{}.png'.format(prefix, par))
    plt.clf()


if __name__ == '__main__':
    main()

