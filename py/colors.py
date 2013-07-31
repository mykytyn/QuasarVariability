import numpy as np
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hdulist = pyfits.open('quasar.fits')


table = hdulist[1].data

#588015509824602234
mask = table['headobjid']==588015509825912905
table = table[mask]

i_band = table['psfMag_i']
i_band_err = table['psfMagerr_i']
mjd_i = table['mjd_i']
r_band = table['psfMag_r']
r_band_err = table['psfMagerr_r']
mjd_r = table['mjd_r']
g_band = table['psfMag_g']
g_band_err = table['psfMagerr_g']
mjd_g = table['mjd_g']

gr_err = np.sqrt(g_band_err**2+r_band_err**2)
ri_err = np.sqrt(r_band_err**2+i_band_err**2)

plt.errorbar(mjd_i,i_band,yerr=i_band_err,marker='.',ls='none')
plt.errorbar(mjd_r,r_band,yerr=r_band_err,marker='.',ls='none')

plt.xlabel('g and i time')
plt.ylabel('g and r mags')
plt.savefig('mag_time2.png')
plt.clf()

plt.errorbar(g_band-r_band,r_band-i_band,xerr=gr_err,yerr=ri_err,marker='.',ls='none')
plt.xlabel('g-r')
plt.ylabel('r-i')
plt.savefig('colorcomp2.png')
plt.clf()

plt.errorbar(g_band,g_band-r_band,xerr=g_band_err,yerr=gr_err,marker='.',ls='none')
plt.xlabel('g')
plt.ylabel('g-r')
plt.savefig('magcolorcomp2.png')


