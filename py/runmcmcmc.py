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
    num_runs = 2
    num_steps = 100
    prefix = 'walker_test'
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
    init_means, init_amps = utils.grid_search_all_bands(data.get_mags(),data.get_sigmas(),data.get_bands())
    initp0 = []
    initp0.extend(init_means)
    initp0.extend((init_amps[2], -1., 50., -1., 0., -1.))
    print initp0

    labels = ['mean u', 'mean g', 'mean r', 'mean i', 'mean z', 'ln a_r', 'alpha', 'ln tau_r', 'beta', 'ln delta_r', 'gamma', 'ln_prob']
    p0 = initp0
    bests = []
    bestsln = []
    bests.append(p0)
    for j in range(num_runs):
        newprefix = '{}-{}'.format(prefix, j)
        p0, pln = run_mcmc(data, newprefix, num_steps, p0, newCovar=True)
        os.rename('{}/{}-triangle.png'.format(init_path, newprefix), '{}/{}-triangle.png'.format(final_path, newprefix))
        for i,par in enumerate(labels):
            os.rename('{}/{}-walker-{}.png'.format(init_path, newprefix, par),'{}/{}-walker-{}.png'.format(final_path, newprefix, par))
        bests.append(p0)
        bestsln.append(pln)

    print bests
    print bestsln
    break

