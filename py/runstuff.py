import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from QuasarVariability import run_mcmc
import utils
import os
import stripe82
import cProfile
import pstats

for i,obj in enumerate(open('targets.txt')):
    obj = int(obj)
    if obj != 587730845813965270:
        continue
    print obj, i
    num_runs = 5
    num_steps = 512
    nthreads = 1
    nwalkers = 32
    prefix = 'bad_data_test'
    init_path = '/home/dwm261/QuasarVariability/py'
    final_path = '/home/dwm261/public_html/Quasars/{}'.format(obj)

    try:
        os.mkdir('{}'.format(final_path))
    except:
        pass

    data = stripe82.Stripe82(obj)
    #utils.make_data_plots(data, prefix='{}-data'.format(prefix))
    #os.rename('{}/{}-data.png'.format(init_path,prefix),'{}/{}-data.png'.format(final_path, prefix))

    data.remove_bad_data('u',22, 26,53500,54000)
    data.remove_bad_data('g',21.,23.,53500,54000)
    data.remove_bad_data('i',20.5,22.,53500,54000)
    data.remove_bad_data('r',20.5,22,53500,54000)
    data.remove_bad_data('z',20.,22,53500,54000)

    data.remove_bad_data('u',19.0,19.6,53800,54200)
    data.remove_bad_data('g',19.0,19.4,53800,54200)
    data.remove_bad_data('r',19.6,19.8,53800,54200)
    data.remove_bad_data('i',19.0,19.15,53800,54200)
    #data.remove_bad_data('z',19.0,19.15,53800,54200)

    data.remove_bad_data('u',22,28,54250,55000)
    data.remove_bad_data('g',19.8,28,54250,55000)
    data.remove_bad_data('r',19.8,28,54250,55000)
    data.remove_bad_data('i',19.7,28,54250,55000)
    data.remove_bad_data('z',19.5,28,54250,55000)

    utils.make_data_plots(data, prefix='{}-data-mod2'.format(prefix))
    os.rename('{}/{}-data-mod2.png'.format(init_path,prefix),'{}/{}-data-mod2.png'.format(final_path, prefix))




    #init_means, init_amps = utils.grid_search_all_bands(data.get_mags(),data.get_sigmas(),data.get_bands())
    #TEMPORARY FOR TESTING:
    init_means = [19.95, 19.75, 19.45, 19.45, 19.35]
    init_amps = [0.08660254, 0.25, 0.12499, 0.1161895, 0.02236068]
    initp0 = []
    initp0.extend(init_means)
    initp0.extend((np.log(init_amps[2]), -1., np.log(100.), 0.))
    onofflist = [True, True, True, True, False]

    p0 = initp0
    bests = []
    bestsln = []
    bests.append(p0)
    default = []
    default.extend(init_means)
    default.extend((np.log(init_amps[2]), -1., np.log(100.), 0., 1.))
    for j in range(num_runs):
        newprefix = '{}-{}'.format(prefix, j)
        sampler, labels, pos, prob, state = run_mcmc(data, num_steps, p0, default, onofflist, newCovar=True, nthreads=nthreads, nwalkers=nwalkers)
        utils.make_triangle_plot(sampler, labels).savefig('%s-triangle.png' % newprefix)
        os.rename('{}/{}-triangle.png'.format(init_path, newprefix), '{}/{}-triangle.png'.format(final_path, newprefix))
        p0, pln = utils.get_best_lnprob(sampler)
        print pln
        walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
        for par,plot in zip(labels,walker_plots):
            plot.savefig('%s-walker-%s.png' % (newprefix, par))
            os.rename('{}/{}-walker-{}.png'.format(init_path, newprefix, par),'{}/{}-walker-{}.png'.format(final_path, newprefix, par))
        bests.append(p0)
        bestsln.append(pln)

    pos, prob, state = sampler.run_mcmc(pos, num_steps)
    print sampler.acor
    acors = []
    count = 1
    while num_steps*count < 64*np.median(sampler.acor):
        print 64*np.median(sampler.acor), num_steps*count
        acors.append(sampler.acor)
        count += 1
        pos, prob, state = sampler.run_mcmc(pos, num_steps)
    print count
    utils.make_triangle_plot(sampler, labels).savefig('%s-triangle.png' % prefix)
    os.rename('{}/{}-triangle.png'.format(init_path, prefix), '{}/{}-triangle.png'.format(final_path, prefix))
    walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
    for par,plot in zip(labels,walker_plots):
        plot.savefig('%s-walker-%s.png' % (prefix, par))
        os.rename('{}/{}-walker-{}.png'.format(init_path, prefix, par),'{}/{}-walker-{}.png'.format(final_path, prefix, par))
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
    plt.savefig('{}-acor.png'.format(prefix))
    os.rename('{}/{}-acor.png'.format(init_path, prefix), '{}/{}-acor.png'.format(final_path, prefix))
    print pos
    print prob
    break

