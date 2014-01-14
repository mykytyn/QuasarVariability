import matplotlib
matplotlib.use('Agg')

from QuasarVariability import run_mcmc
from utils import make_data_plots
import os
import stripe82

for i,obj in enumerate(open('targets.txt')):
    obj = int(obj)
    print obj
    prefix = 'new-{}'.format(obj)
    #prefix2 = 'alpha-notau-{}'.format(obj)
    init_path = '/home/dwm261/QuasarVariability/py'
    final_path = '/home/dwm261/public_html/Quasars/{}'.format(obj)
    if i==0:
        continue
    make_data_plots(stripe82.Stripe82(obj), prefix='{}-data'.format(prefix))
    run_mcmc(obj,'new',newCovar=True)
    #run_mcmc(obj,'alpha-notau',noTau=True,alpha=True)
    try:
        os.mkdir('{}'.format(final_path))
    except:
        pass
    os.rename('{}/{}-triangle.png'.format(init_path, prefix), '{}/{}-triangle.png'.format(final_path, prefix))
    #os.rename('{}/{}-triangle.png'.format(init_path, prefix2), '{}/{}-triangle.png'.format(final_path, prefix2))
    os.rename('{}/{}-data.png'.format(init_path,prefix),'{}/{}-data.png'.format(final_path, prefix))
    #os.rename('{}/{}-posterior.png'.format(init_path, prefix), '{}/{}-posterior.png'.format(final_path, prefix))
    #os.rename('{}/{}-posterior.png'.format(init_path, prefix2), '{}/{}-posterior.png'.format(final_path, prefix2))

    for i in range(11):
        os.rename('{}/{}-walker-dim{}.png'.format(init_path, prefix, i),'{}/{}-walker-dim{}.png'.format(final_path, prefix, i))
    #for i in range(7):
        #os.rename('{}/{}-walker-dim{}.png'.format(init_path, prefix2, i),'{}/{}-walker-dim{}.png'.format(final_path, prefix2, i))

    break
