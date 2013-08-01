import numpy as np
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lnprob import *

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


for g,gt,r,rt,g_err,r_err in zip(g_band,mjd_g,r_band,mjd_r,g_band_err,r_band_err):
    assert np.abs(gt-rt)<1./24

#newmask = [np.abs(mjd_g-53697.)>1.]
print np.abs(mjd_g-53697.)
#print np.all(newmask)
#print newmask
#table = table[newmask]
g_band = table['psfMag_g']
g_band_err = table['psfMagerr_g']
mjd_g = table['mjd_g']
r_band = table['psfMag_r']
r_band_err = table['psfMagerr_r']
mjd_r = table['mjd_r']
s2 = 0.
b = .2

az = []
probs = []

for a in np.arange(0.,2.,.01):
    az.append(a)
    probs.append(np.sum(ln_like_array(g_band,r_band,g_band_err**2,r_band_err**2,a,b,s2)))

best_index = np.argmax(probs)
best = az[best_index]

plt.plot(az,probs,'.')
plt.savefig('proba2.png')
print best
plt.clf()

plt.errorbar(mjd_g,g_band,marker='.',ls='none',color='g')
plt.errorbar(mjd_r,r_band,marker='.',ls='none',color='r')
plt.errorbar(mjd_g,(g_band-.3)/best,marker='.',ls='none',color='k')
plt.ylim(18.0,20.5)
plt.savefig('besta.png')






