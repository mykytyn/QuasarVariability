import numpy as np
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rmag_utils import *
import os

hdulist = pyfits.open('quasar.fits')

table = hdulist[1].data
#print table.columns.names

#run6417
mask = table['headobjid']==588015509825912905
table = table[mask]

r_band = table['psfMag_r']
r_band_err = table['psfMagerr_r']
r_band_var=r_band_err**2
mjd_r = table['mjd_r']
mean_r = np.mean(r_band)
std_r = np.std(r_band)
r_var= std_r**2

plt.figure(1)
plt.errorbar(mjd_r,r_band,yerr=r_band_err,marker='.',ls='none')
plt.axhline(y=mean_r)
plt.xlabel('time(days)')
plt.ylabel('r magnitude')
plt.savefig('mag_time.png')

plt.figure(2)
r_var=1
mean_r=[]
probs=[]
for i in np.arange(10,30,.1):
    mean_r.append(i)
    probs.append(np.sum(ln_likelihood(r_band, r_band_var, i, r_var)))

plt.plot(mean_r, probs, 'r.')
plt.xlabel('mean r-mag')
plt.ylabel('sum of likelihoods')
plt.savefig('mean_tot_like.pdf')
os.system('cp mean_tot_like.pdf ~/public_html/QSO')

plt.figure(3)
mean_r=max(mean_r)
r_var=[]
probs=[]
for i in np.arange(0,.05,.001):
    r_var.append(i)
    probs.append(np.sum(ln_likelihood(r_band, r_band_var, mean_r, i)))
plt.plot(r_var,probs,'r.')
plt.xlim(-.02,.06)
plt.ylabel('sum of likelihoods')
plt.xlabel(r'$\Sigma^2$')
plt.savefig('rvar_tot_like.pdf')
os.system('cp rvar_tot_like.pdf ~/public_html/QSO')


# set up arrays
meanmin = 10.
meanmax = 30.
sigmamin = 0.
sigmamax = .05
meanstep = .1
sigmastep = .001
sigmas = np.arange(sigmamin,sigmamax,sigmastep)
means = np.arange(meanmin,meanmax,meanstep)
mm,ss = np.meshgrid(means,sigmas)
probs = np.zeros_like(mm)
print len(probs)

# do stupid loop
s=0
ni,nj = mm.shape
print ni,nj
for i in range(ni):
    for j in range(nj):
        pars = np.array([mm[i,j],ss[i,j]])
        probs[i,j] = tot_likelihood(pars, r_band, r_band_var)
print probs
plt.clf()
vmax = np.max(probs)
vmin = vmax-1000.
plt.imshow(probs,cmap='gray',interpolation='nearest',origin='lower',
           extent=[meanmin,meanmax,sigmamin,sigmamax+.05], vmax=vmax,vmin=vmin,aspect='auto')
plt.colorbar()
plt.savefig('grid.png')
