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
import cPickle

def run_all(objid, path, prefix, pool, cutoff=.2, S=False):#rename S
    obj = int(objid)   
    num_steps = 128
    nwalkers = 32
    quasar_data = data.Stripe82(obj)

    quasar_data.remove_bad_data(cutoff,2./24.)
    utils.make_data_plots(quasar_data).savefig('{}/{}-data.png'.format(path, prefix))

    print "starting grid search"
    init_means, init_amps = utils.grid_search_all_bands(quasar_data.get_mags(),quasar_data.get_sigmas(),quasar_data.get_bands()) #reminder, look at this func
    #a_r, alpha, tau, delta_r, gamma, S
    if S:
        onofflist = [True, True, True, True, False, True]
    else:
        onofflist = [False, True, False, True, False, False]
    default = []
    default.extend(init_means)
    #ln a_r, alpha, ln tau, delta_r, gamma ,S
    default.extend((np.log(init_amps[2]), -1., np.log(100.), 0., 1., 0.))
    default = np.array(default)
    params = np.array(default[5:])
    params = params[np.array(onofflist)]
    p0 = np.concatenate([default[:5], params])

    print "starting mcmc burn in"
    sampler, labels, pos, prob, state = qv.run_mcmc(quasar_data, num_steps, default, onofflist, pool=pool, nwalkers=nwalkers)

    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, num_steps)
    acors = []
    count = 1

    print "running mcmc"
    while num_steps*count < 64*np.median(sampler.acor) and count<1: #this can be better i think
        print 64*np.median(sampler.acor), num_steps*count
        acors.append(sampler.acor)
        count += 1
        pos, prob, state = sampler.run_mcmc(pos, num_steps)
    print "triangle plot"
    utils.make_triangle_plot(sampler, labels).savefig('{}/{}-triangle.png'.format(path, prefix))
    print "walker plots"
    walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
    for par,plot in zip(labels,walker_plots):
        plot.savefig('{}/{}-walker-{}.png'.format(path, prefix, par))
    plt.clf()

    #MOVE THIS INTO UTILS, PLOT
    #steps_plot = range(len(acors))
    #steps_plot = [num_steps*(x+1) for x in steps_plot]

    #for j,par in enumerate(labels):
        #if par == "ln_prob":
            #continue
        #plt.plot(steps_plot,[x[j] for x in acors], label='%s' % par)

    #plt.xlabel('Number of steps')
    #plt.ylabel('acor')
    #plt.legend(loc='upper left')
    #plt.savefig('{}/{}-acor.png'.format(path,prefix))

    bestparams, bestprob = utils.get_best_lnprob(sampler)

    """oldtimes = quasar_data.get_times(bandname='r')
    oldbad = quasar_data.get_bad_mask()
    deltalns = []
    for time in oldtimes:
        quasar_data.remove_point(time)
        mags, times, bands, sigmas, bads = quasar_data.get_data()
        deltaln = qv.temp_ln(bestparams, mags, times, bands, sigmas, default, onofflist)-bestprob
        print deltaln
        deltalns.append(deltaln)
        print len(deltalns)
        quasar_data.bad = oldbad
    """
    quasar = qv.QuasarVariability(qv.RandomWalk(default[5:], onofflist, qv.wavelengths, qv.base), default[0:5])
    quasar.unpack_pars(bestparams)
    #f = open('test.pickle', 'w')
    #cPickle.dump(quasar, f)
    #f.close()
    print "Making posteriors/IRLS"
    for i in range(5):
        #priorplot.savefig('{}/{}-prior.png'.format(path,prefix))
        #posteriorplot = utils.make_posterior_plots(quasar, quasar_data, deltalns=deltalns, num_samps=0)
        #posteriorplot.savefig('{}/{}-{}-bad-posterior.png'.format(path,prefix,i))
        posteriorplot = utils.make_posterior_plots(quasar, quasar_data, deltalns=None, num_samps=8)
        posteriorplot.savefig('{}/{}-{}-posterior.png'.format(path,prefix,i))
        quasar_data.IRLS_update_sigmas(quasar)


    return bestparams, bestprob


if __name__== '__main__':
    #pool = Pool(10)
    pool = None
    cutoff = 100
    bpoint = True
    for obj in ('587730845813768280',):#put in bad one
    #alpha of point based on delta-ln-like
        obj = int(obj)
        path = '/home/dwm261/public_html/Quasars/newkernel/{}'.format(obj)
        print path
        try:
            os.mkdir(path)
        except:
            pass
        #best, bestprob = run_all(obj, path, 'newkernel-{}'.format(cutoff), pool, cutoff, S=True)
        best, bestprob = run_all(obj, path, 'temp-{}'.format(cutoff), pool, cutoff, S=False)
        result = [obj, cutoff, bestprob]
        result.extend(best)
        print result
        assert False
