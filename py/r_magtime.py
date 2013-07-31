import numpy as np
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rmag_utils import *
hdulist = pyfits.open('quasar.fits')

table = hdulist[1].data
print table.columns.names

print table['psfSigma1_r'][0], table['psfSigma2_r'][0]
#run6417
mask = table['headobjid']==588015509825912905
table = table[mask]
print len(table)

r_band = table['psfMag_r']
r_band_err = table['psfMagerr_r']
r_band_var=r_band_err**2
mjd_r = table['mjd_r']
mean_r = np.mean(r_band)
std_r = np.std(r_band)
r_var= std_r**2

plt.figure()
plt.errorbar(mjd_r,r_band,yerr=r_band_err,marker='.',ls='none')
plt.axhline(y=mean_r)
plt.xlabel('time(days)')
plt.ylabel('r magnitude')
plt.savefig('mag_time.png')

print ln_1d_gauss(r_band, mean_r,r_band_var)
print ln_likelihood(r_band, r_band_var, mean_r, r_var)

total_ln_likelihood=np.sum(ln_likelihood(r_band, r_band_var, mean_r, r_var))

print total_ln_likelihood
