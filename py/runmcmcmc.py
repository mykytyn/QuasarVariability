import numpy as np
import matplotlib
matplotlib.use('Agg')

from QuasarVariability import run_mcmc
import utils
import os
import stripe82

for i,obj in enumerate(open('targets.txt')):
    obj = int(obj)
    print obj
    num_runs = 4
    num_steps = 256
    nthreads = 15
    nwalkers = 32
    prefix = 'ekta_test'
    #prefix2 = 'alpha-notau-{}'.format(obj)
    init_path = '/home/dwm261/QuasarVariability/py'
    final_path = '/home/dwm261/public_html/Quasars/{}'.format(obj)
    if i==0:
        continue
    utils.make_data_plots(stripe82.Stripe82(obj), prefix='{}-data'.format(prefix))

    try:
        os.mkdir('{}'.format(final_path))
    except:
        pass
    os.rename('{}/{}-data.png'.format(init_path,prefix),'{}/{}-data.png'.format(final_path, prefix))
    data = stripe82.Stripe82(obj)
    #init_means, init_amps = utils.grid_search_all_bands(data.get_mags(),data.get_sigmas(),data.get_bands())
    #TEMPORARY FOR TESTING:
    init_means = [19.95, 19.75, 19.45, 19.45, 19.35]
    init_amps = [0.08660254, 0.25, 0.12499, 0.1161895, 0.02236068]
    initp0 = []
    initp0.extend(init_means)
    initp0.extend((np.log(init_amps[2]), -1., np.log(50.), 0., -1.))
    print initp0
    onofflist = [True, True, True, False, True, True]

    p0 = initp0
    bests = []
    bestsln = []
    bests.append(p0)
    default = []
    default.extend(init_means)
    default.extend((np.log(init_amps[2]), -1., np.log(50.), 0., 0., -1.))
    for j in range(num_runs):
        newprefix = '{}-{}'.format(prefix, j)
        sampler, labels, pos, prob, state = run_mcmc(data, num_steps, p0, default, onofflist, newCovar=True, nthreads=nthreads, nwalkers=nwalkers)
        #utils.make_triangle_plot(sampler, labels).savefig('%s-triangle.png' % newprefix)
        p0, pln = utils.get_best_lnprob(sampler)
        #walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
        #os.rename('{}/{}-triangle.png'.format(init_path, newprefix), '{}/{}-triangle.png'.format(final_path, newprefix))
        #for par,plot in zip(labels,walker_plots):
            #plot.savefig('%s-walker-%s.png' % (newprefix, par))
            #os.rename('{}/{}-walker-{}.png'.format(init_path, newprefix, par),'{}/{}-walker-{}.png'.format(final_path, newprefix, par))
        bests.append(p0)
        bestsln.append(pln)

    pos, prob, state = sampler.run_mcmc(pos, 256)
    print sampler.acor
    print 32.*np.max(sampler.acor)
    while 256 < 32.*np.max(sampler.acor):
        print sampler.acor
        print 32.*np.max(sampler.acor)
        pos, prob, state = sampler.run_mcmc(pos,256)
    utils.make_triangle_plot(sampler, labels).savefig('%s-triangle.png' % prefix)
    os.rename('{}/{}-triangle.png'.format(init_path, prefix), '{}/{}-triangle.png'.format(final_path, prefix))
    walker_plots = utils.make_walker_plots(sampler, labels, nwalkers)
    for par,plot in zip(labels,walker_plots):
        plot.savefig('%s-walker-%s.png' % (prefix, par))
        os.rename('{}/{}-walker-{}.png'.format(init_path, prefix, par),'{}/{}-walker-{}.png'.format(final_path, prefix, par))

    print pos
    print prob
    break

