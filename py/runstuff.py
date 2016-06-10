import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import QuasarVariability as qv
import utils
import os
import sys
import data
from multiprocessing import Pool
import emcee
import cPickle


def make_default(quasar_data, init_tau = 100.):
    """
    Generates default parameters for start of MCMC and for IRLS
    todo: make these adjustable and less magic perhaps?
    optimization?
    """
    default = []
    for b in ['u', 'g', 'r', 'i', 'z']:
        default.append(np.median(quasar_data.get_mags(bandname=b)))
    init_amp = .2 #magic start guess for amp
    init_alpha = -1 #magic start guess for alpha
    init_delta_r = 0.
    init_gamma = 1.
    init_S = 0.
    #ln a_r, alpha, ln tau, delta_r, gamma ,S
    default.extend((np.log(init_amp), init_alpha, np.log(init_tau), init_delta_r, init_gamma, init_S))
    default = np.array(default)
    return default

def initialize_mcmc(quasar_data, default, onofflist,
                   pool, nwalkers=16, quasar=None):
    """
    Generates a emcee sampler object and initial position of walkers
    todo: probably does not need its own function, can be moved into an other
    """
    print "initializing sampler"
    mags, times, bands, sigmas, bad = quasar_data.get_data()
    bandnames = quasar_data.get_bandlist()
    
    params = np.array(default[5:])
    params = params[np.array(onofflist)]
    p0 = np.concatenate([default[:5], params])

    if quasar is None:
        quasar = qv.QuasarVariability(qv.RandomWalk(default[5:], onofflist,
                                                    qv.wavelengths, qv.base), default[:5])

    labels = quasar.get_labels()
    ndim = len(labels)
    labels.append('ln_prob')

    initial = []
    for i in range(nwalkers):  # could probably be improved -mykytyn
        pp = p0 + 0.0001 * np.random.normal(size=len(p0))  # Magic number
        initial.append(pp)
    arguments = [mags, times, bands, sigmas, default, onofflist]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, qv.temp_ln,
                                    args=arguments, pool=pool)

    return sampler, labels, initial


def burn_in(sampler, num_steps, initial):
    """
    Runs a "burn-in" for specified number of steps starting from initial position given
    """
    print "starting mcmc burn in"
    pos, prob, state = sampler.run_mcmc(initial, num_steps)
    sampler.reset()
    return pos, prob, state



def main_mcmc(sampler, pos, stepsize, acorsteps, maxcount):
    """
    Runs emcee until either a max is returned or the acor has become small enough compared to the steps
    """
    acors = []
    count = 0

    print "running mcmc"
    pos, prob, state = sampler.run_mcmc(pos, stepsize)
    count = count+1
    acors.append(sampler.acor)
    while stepsize*count < acorsteps*np.median(sampler.acor) and count<maxcount: #this can be better i think
        print acorsteps*np.median(sampler.acor), stepsize*count
        acors.append(sampler.acor)
        count += 1
        pos, prob, state = sampler.run_mcmc(pos, stepsize)

    return pos, prob, state

def triangle_and_walker(sampler, labels, path, prefix, nwalkers):
    """
    Generates and saves the triangle and walker plots for the outcome of an mcmc process
    """
    print "triangle plot"
    utils.make_triangle_plot(sampler.lnprobability, sampler.flatchain, labels).savefig('{}/{}-triangle.png'.format(path, prefix))
    #print "walker plots"
    #walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
    #for par,plot in zip(labels,walker_plots):
        #plot.savefig('{}/{}-walker-{}.png'.format(path, prefix, par))
    #plt.clf()

def acor_plot(): #TEMP TEMP TEMP skeleton does not work! TEMP
    """ 
    Does not work, should give acor for each parameter by number of steps
    p.s. make this object-oriented etc., move to utils
    """
    steps_plot = range(len(acors))
    steps_plot = [num_steps*(x+1) for x in steps_plot]

    for j,par in enumerate(labels):
        if par == "ln_prob":
            continue
        plt.plot(steps_plot,[x[j] for x in acors], label='%s' % par)

    plt.xlabel('Number of steps')
    plt.ylabel('acor')
    plt.legend(loc='upper left')
    plt.savefig('{}/{}-acor.png'.format(path,prefix))


def run_irls(quasar, quasar_data, Q=5, num_samps=0, repeats=5):
    """
    Updates sigmas according to IRLS
    """
    for i in range(repeats):
        print "UPDATING SIGMAS {}".format(i)
        quasar_data.IRLS_update_sigmas(quasar, Q=Q)

def main_sequence(obj, prefix, path, init_tau, IRLS_Q):
    #path = '{}/{}'.format(path,obj)
    print path, prefix
    save_plots = path is not None
    if save_plots:
        try:
            os.mkdir(path)
        except:
            pass
    quasar_data = data.Stripe82(obj)
    if save_plots:
        utils.make_data_plots(quasar_data).savefig('{}/{}-data.png'.format(path, prefix))

    default = make_default(quasar_data,init_tau=init_tau)
    
    quasar = qv.QuasarVariability(qv.RandomWalk(default[5:], onofflist, qv.wavelengths, qv.base), default[0:5])
    if save_plots:
        utils.make_posterior_plots(quasar, quasar_data, num_samps=0).savefig('{}/{}-1-orig-posterior.png'.format(path,prefix))
    run_irls(quasar,quasar_data, Q=IRLS_Q, repeats=10)
    if save_plots:
        utils.make_posterior_plots(quasar, quasar_data, num_samps=0).savefig('{}/{}-1-final-posterior.png'.format(path,prefix))
    
    sampler, labels, initial = initialize_mcmc(quasar_data, default, onofflist, pool, nwalkers, quasar=None)
    pos, prob, state = burn_in(sampler, num_steps, initial)
    pos, prob, state = main_mcmc(sampler, pos, stepsize, acorsteps, maxcount)
    if save_plots:
        triangle_and_walker(sampler, labels, path, prefix, nwalkers)
    bestparams, bestprob = utils.get_best_lnprob(sampler)
    quasar.unpack_pars(bestparams)
    f = open("{}.pickle".format(prefix),'w')
    cPickle.dump([quasar, quasar_data, sampler.flatchain, sampler.lnprobability, labels], f)
    f.close()
    print bestparams
    if save_plots:
        utils.make_posterior_plots(quasar, quasar_data, num_samps=0).savefig('{}/{}-postfit-posterior.png'.format(path,prefix))


if __name__== '__main__':
    pool = Pool(10)
    #pool = None
    #a_r, alpha, tau, delta_r, gamma, S
    onofflist = [True, True, True, False, False, False]
    num_steps = 2048
    nwalkers = 32
    stepsize = 256
    maxcount = 50
    acorsteps = 64
    init_taus = [50.,100.]
    IRLS_Q = 2.

    for init_tau in init_taus:
        f = open('newtargetlist.txt', 'r')
        for numobj,obj in enumerate(f):
            obj = int(obj)
            prefix = '{}tau-Q2-{}'.format(int(init_tau), obj)
            if os.path.exists('{}.pickle'.format(prefix)):
                continue
            path = None
            main_sequence(obj, prefix, path, init_tau=init_tau,IRLS_Q=IRLS_Q)
        f.close()
