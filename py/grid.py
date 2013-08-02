import numpy as np
from lnprob import ln_prob
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def make_grid_plots(objid):

    hdulist = pyfits.open('quasar.fits')

    
    fulltable = hdulist[1].data
    mask = fulltable['headobjid']==objid
    table = fulltable[mask]
    print len(table)

    mjd_r = table['mjd_r']

    newmask = [np.abs(mjd_r-53697.)>1.]
    print np.abs(mjd_r-53697.)
    #print np.all(newmask)
    #print newmask
    table = table[newmask]
    r_band = table['psfMag_r']
    r_band_err = table['psfMagerr_r']
    mjd_r = table['mjd_r']
    r_mean = np.mean(r_band)

    for target in ['u','g','r','i','z']:
        target_band = table['psfMag_%s' % target]
        target_band_err = table['psfMagerr_%s' %target]
        mjd_target = table['mjd_%s' %target]

        # set up arrays
        bmin = np.mean(target_band)-1.
        bmax = np.mean(target_band)+1.
        amin = .3
        amax = 1.4
        astep = .01
        bstep = .01
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
                pars = np.array([aa[i,j],bb[i,j],s,r_mean])
                probs[i,j] = ln_prob(pars,target_band,r_band,target_band_err,r_band_err)
    
        plt.clf()
        vmax = np.max(probs)
        vmin = vmax-10.
        plt.imshow(probs,cmap='gray',interpolation='nearest',origin='lower',
                   extent=[amin,amax,bmin,bmax], vmax=vmax,vmin=vmin,aspect='auto')
        plt.colorbar()
        plt.title('Objid: %d, Band: %s' % (objid,target))
        plt.xlabel('a param')
        plt.ylabel('b param')
        plt.savefig('grid-%d-%s.png' % (objid,target))
        loc = np.argmax(probs)
        bloc,aloc = np.unravel_index(loc,probs.shape)
        print aloc,bloc
        a = az[aloc]
        b = bs[bloc]
        print a,b
    
        plt.clf()

        plt.errorbar(mjd_target,target_band,marker='.',ls='none',color='g',
                     label='Band: %s' % target)
        plt.errorbar(mjd_r,r_band,marker='.',ls='none',color='r',label='Band: r')
        plt.errorbar(mjd_target,(target_band-b)/a+r_mean,marker='.',ls='none',color='k',
                     label='a=%g, b=%g' % (a,b))
        plt.xlabel('Time (days)')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('Objid: %d' % objid)
        #plt.ylim(18.0,20.5)
        plt.savefig('grid-%d-%s-overlay.png' % (objid,target))




