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
print r_band_err,g_band_err

# set up arrays
bmin = -.995
bmax = 1.
amin = 0.005
amax = 2.
astep = .001
bstep = .001
bs = np.arange(bmin,bmax,bstep)
az = np.arange(amin,amax,astep)
aa,bb = np.meshgrid(az,bs)
probs = np.zeros_like(aa)

# do stupid loop
s=0
ni,nj = aa.shape
print ni,nj
for i in range(ni):
    for j in range(nj):
        pars = np.array([aa[i,j],bb[i,j],s])
        probs[i,j] = ln_prob(pars,g_band,r_band,g_band_err,r_band_err)
print probs
plt.clf()
vmax = np.max(probs)
vmin = vmax-100.
plt.imshow(probs,cmap='gray',interpolation='nearest',origin='lower',
           extent=[amin,amax,bmin,bmax], vmax=vmax,vmin=vmin,aspect='auto')
plt.colorbar()
plt.savefig('grid2.png')
loc = np.argmax(probs)
bloc,aloc = np.unravel_index(loc,probs.shape)
print aloc,bloc
a = az[aloc]
b = bs[bloc]
print a,b

plt.clf()

plt.errorbar(mjd_g,g_band,marker='.',ls='none',color='g')
plt.errorbar(mjd_r,r_band,marker='.',ls='none',color='r')
plt.errorbar(mjd_g,(g_band-b)/a,marker='.',ls='none',color='k')
plt.ylim(18.0,20.5)
plt.savefig('bestgrid2.png')




