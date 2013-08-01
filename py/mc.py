import numpy as np
import emcee
import pyfits
import random
from lnprob import ln_prob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


objid = 587730845813768280

hdulist = pyfits.open('quasar.fits')


fulltable = hdulist[1].data
mask = fulltable['headobjid']==objid
table = fulltable[mask]
print len(table)


g_band = table['psfMag_g']
g_band_err = table['psfMagerr_g']
mjd_g = table['mjd_g']
r_band = table['psfMag_r']
r_band_err = table['psfMagerr_r']
mjd_r = table['mjd_r']


ndim = 3
args = g_band,r_band,g_band_err,r_band_err

nwalkers = 100
p0 = []
for i in range(nwalkers):
    p0.append([random.uniform(.5,1.5),random.uniform(-.5,.5),random.uniform(-.5,.5)])


sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_prob,args=[g_band,r_band,g_band_err,r_band_err])

pos,prob,state = sampler.run_mcmc(p0,100)
sampler.reset()
pos,prob,state = sampler.run_mcmc(pos,1000)

for i in range(ndim):
    plt.clf()
    plt.hist(sampler.flatchain[:,i],100,color='k',histtype="step")
    plt.title("Dimension {0:d}".format(i))
    plt.savefig('testmcmc-%d' % i)

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

