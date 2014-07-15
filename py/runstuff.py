import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from QuasarVariability import run_mcmc
import utils
import os
import data
from prior_posterior import graph_prior_and_posterior

def run_all(objid, path, prefix, cutoff=.2):
    obj = int(objid)   
    num_runs = 5
    num_steps = 512
    nthreads = 10
    nwalkers = 32
    quasar_data = data.Stripe82(obj)

    quasar_data.remove_bad_data(cutoff,2./24.)    
    utils.make_data_plots(quasar_data).savefig('{}/{}-data.png'.format(path, prefix))

    init_means, init_amps = utils.grid_search_all_bands(quasar_data.get_mags(),quasar_data.get_sigmas(),quasar_data.get_bands())
    onofflist = [True, True, True, True, False]
    default = []
    default.extend(init_means)
    default.extend((np.log(init_amps[2]), -1., np.log(100.), 0., 1.))
    default = np.array(default)
    params = np.array(default[5:])
    params = params[np.array(onofflist)]
    print params
    p0 = np.concatenate([default[:5], params])
    print p0

    for j in range(num_runs):
        newprefix = '{}-{}'.format(prefix, j)
        sampler, labels, pos, prob, state = run_mcmc(quasar_data, num_steps, default, onofflist,  nthreads=nthreads, nwalkers=nwalkers)

    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, num_steps)
    print sampler.acor
    acors = []
    count = 1

    while num_steps*count < 64*np.median(sampler.acor) and count<20:
        print 64*np.median(sampler.acor), num_steps*count
        acors.append(sampler.acor)
        count += 1
        pos, prob, state = sampler.run_mcmc(pos, num_steps)
    print count
    utils.make_triangle_plot(sampler, labels).savefig('{}/{}-triangle.png'.format(path, prefix))
    walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
    for par,plot in zip(labels,walker_plots):
        plot.savefig('{}/{}-walker-{}.png'.format(path, prefix, par))
    print acors
    plt.clf()
    steps_plot = range(len(acors))
    steps_plot = [num_steps*(x+1) for x in steps_plot]

    print steps_plot
    for j,par in enumerate(labels):
        if par == "ln_prob":
            continue
        print [x[j] for x in acors]
        plt.plot(steps_plot,[x[j] for x in acors], label='%s' % par)

    plt.xlabel('Number of steps')
    plt.ylabel('acor')
    plt.legend(loc='upper left')
    plt.savefig('{}/{}-acor.png'.format(path,prefix))

    bestparams, bestprob = utils.get_best_lnprob(sampler)
    priorplot, posteriorplot = graph_prior_and_posterior(quasar_data, bestparams, onofflist, default, prefix=prefix)
    priorplot.savefig('{}/{}-prior.png'.format(path,prefix))
    posteriorplot.savefig('{}/{}-posterior.png'.format(path,prefix))

    print pos
    print prob
    return bestparams, bestprob


if __name__== '__main__':
    #f = open('results2.txt', 'w')
    for cutoff in [.2]:
        for i,obj in enumerate(open('newlist.txt')):
            obj = int(obj)            
            path = '/home/dwm261/public_html/Quasars/new/{}'.format(obj)
            print path
            try:
                os.mkdir(path)
            except:
                pass
            best, bestprob = run_all(obj, path, 'posdelta-{}'.format(cutoff),cutoff)
            result = [obj, cutoff, bestprob]
            result.extend(best)
            print result
            #f.write('{}\n'.format(result))
    #f.close()
