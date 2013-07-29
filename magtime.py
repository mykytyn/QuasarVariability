import numpy as np
import pyfits
import matplotlib.pyplot as plt


hdulist = pyfits.open('quasar.fits')


table = hdulist[1].data
print table.columns

print table['headobjid'][::1000]
#run6417
mask = table['headobjid']==588015509825912905
table = table[mask]
print len(table)
assert(False)
i_band = table['psfMag_i']
i_band_err = table['psfMagerr_i']
mjd_i = table['mjd_i']
r_band = table['psfMag_r']
r_band_err = table['psfMagerr_r']
mjd_r = table['mjd_r']

plt.errorbar(mjd_i,i_band,yerr=i_band_err,marker='.',ls='none')
plt.errorbar(mjd_r,r_band,yerr=r_band_err,marker='.',ls='none')
plt.savefig('mag_time.png')

