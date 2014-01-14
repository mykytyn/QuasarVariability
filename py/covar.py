import numpy as np
from lnprob import ln_prob
import pyfits
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

newmask = [np.abs(mjd_g-53697.)>1.]
print np.abs(mjd_g-53697.)
#print np.all(newmask)
#print newmask
table = table[newmask]
g_band = table['psfMag_g']
g_band_err = table['psfMagerr_g']
mjd_g = table['mjd_g']
r_band = table['psfMag_r']
r_band_err = table['psfMagerr_r']
mjd_r = table['mjd_r']
r_mean = np.mean(r_band)
print r_band_err,g_band_err


#ts = np.arange(0,1000.,50.)
ts = mjd_r
a = 1.12
b = 18.6224

for tau in [200]:
    plt.clf()
    tt,tt = np.meshgrid(ts,ts)
    ti,tj = tt.shape

    V = np.zeros_like(tt)

    for i in range(ti):
        for j in range(tj):
            V[i][j] = np.exp(-np.abs(ts[i]-ts[j])/tau)


    mean = np.zeros_like(ts)
#mean = [x for x in range(20)]

    samples = np.random.multivariate_normal(mean,V,5)
    print samples

    for i in range(len(samples)):
        plt.plot(ts,samples[i],'.-')
    plt.title('Tau = %d' % tau)
    plt.savefig('samples-%d.png' % tau)
