import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
import QuasarVariability

inittime = 0.
finaltime = 1000.
dt = 10.

num_samps = 4

tau = 400.
a = .75
bandlist = ['q']

timegrid, bandsgrid = utils.make_band_time_grid(inittime, finaltime, dt, bandlist)

covar = QuasarVariability.RandomWalk(np.array([a]), tau)

quasarobj = QuasarVariability.QuasarVariability(covar, np.array([1.]))

maggrids = []
bands = np.array(['q'])
matplotlib.rc('xtick', labelsize=6)
matplotlib.rc('ytick', labelsize=6)
for i in range(num_samps):
    maggrid = quasarobj.get_prior_sample(timegrid, bandsgrid)
    maggrid = np.array(maggrid).reshape(timegrid.shape)
    maggrids.append(maggrid)


btimegrid = timegrid
pmean = quasarobj.get_mean_vector(timegrid, bandsgrid)
Vpp = quasarobj.get_variance_tensor(timegrid, bandsgrid)
pmean = np.array(pmean).reshape(timegrid.shape)
psig = np.sqrt(np.diag(np.array(Vpp)))

bpmean = pmean
bpsig = psig
plt.plot(btimegrid, bpmean, 'k-')
plt.plot(btimegrid, bpmean-bpsig, 'k-')
plt.plot(btimegrid, bpmean+bpsig, 'k-')
for j in range(num_samps):
    maggrid = maggrids[j]
    plt.plot(btimegrid, maggrid, 'k-', alpha=0.25)

plt.ylabel('q')
plt.savefig('latentvar.png')
