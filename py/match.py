import numpy as np
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

f = open('ids.txt','r')
hdulist = pyfits.open('quasar.fits')


fulltable = hdulist[1].data

for j,x in enumerate(f):
    plt.clf()
    x = int(x)


    mask = fulltable['headobjid']==x
    table = fulltable[mask]
    print len(table)
    if len(table)<50:
        continue
    g_band = table['psfMag_g']
    g_band_err = table['psfMagerr_g']
    mjd_g = table['mjd_g']
    r_band = table['psfMag_r']
    r_band_err = table['psfMagerr_r']
    mjd_r = table['mjd_r']

    for g,gt,r,rt in zip(g_band,mjd_g,r_band,mjd_r):
        if np.abs(gt-rt)<1./24:
            plt.plot((gt,rt),(g,r),'r')
    plt.errorbar(mjd_g,g_band,marker='.',ls='none')
    plt.errorbar(mjd_r,r_band,marker='.',ls='none')
    #plt.xlim(53695,53700)
    plt.savefig('match-%d.png' % j)
    print j,x
    if j>20:
        break

