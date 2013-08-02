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

plt.plot(mean_r, probs, 'r-')
plt.xlabel('mean r-mag')
plt.ylim(np.array([-100, 1] + np.max(probs)))
plt.ylabel('sum of likelihoods')
plt.savefig('mean_tot_like.pdf')
os.system('cp mean_tot_like.pdf ~/public_html/QSO')

plt.figure(3)
mean_r=mean_r[np.argmax(probs)]
r_var=[]
probs=[]
for i in np.arange(0,1.,.001):
    r_var.append(i)
    probs.append(np.sum(ln_likelihood(r_band, r_band_var, mean_r, i)))
plt.plot(r_var,probs,'r-')
print np.max(probs)
plt.ylim(np.array([-100, 1] + np.max(probs)))
plt.ylabel('sum of likelihoods')
plt.xlabel(r'$\Sigma^2$')
plt.savefig('rvar_tot_like.pdf')
os.system('cp rvar_tot_like.pdf ~/public_html/QSO')

# set up arrays
meanmin = 10.
meanmax = 30.
sigmamin = 0.
sigmamax = 1.
meanstep = .1
sigmastep = .001
sigmas = np.arange(sigmamin,sigmamax,sigmastep)
means = np.arange(meanmin,meanmax,meanstep)
mm,ss = np.meshgrid(means,sigmas)
probs = np.zeros_like(mm)

#do stupid loop
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

plt.figure(4)
loc=np.argmax(probs)
sigloc,meanloc=np.unravel_index(loc,probs.shape)
mean_r=means[meanloc]
std=np.sqrt(sigmas[sigloc])

#obj=588015509825912905

fn=open('ids.txt','r')
objs=fn.readlines()
objs=[o.strip('\n') for o in objs]
objs=[int(o) for o in objs]
objs=objs[0:5]
print objs
assert(False)
for obj in objs:
    print obj
    plt.clf()
    print get_mag('u',int(obj)), get_mag_err('u',int(obj))
    plt.errorbar(get_time('r',obj),get_mag('r',obj),yerr=get_mag_err('r',obj),marker='.',ls='none', label='r')
    plt.errorbar(get_time('u',obj),get_mag('u',obj),yerr=get_mag_err('u',obj),marker='.',ls='none',label='u')
    plt.errorbar(get_time('g',obj),get_mag('u',obj),yerr=get_mag_err('g',obj),marker='.',ls='none',label='g')
    plt.errorbar(get_time('i',obj),get_mag('i',obj),yerr=get_mag_err('i',obj),marker='.',ls='none',label='i')
    plt.errorbar(get_time('z',obj),get_mag('z',obj),yerr=get_mag_err('z',obj),marker='.',ls='none',label='z')
    print 'error bars done'

    rband_lines=meshgrid(10., 30., 0., 1., .1, .001, 'r', obj)
    plt.axhline(y=rband_lines[0],color='green')
    plt.axhline(y=rband_lines[0]+rband_lines[1], linestyle='--',color='blue')
    plt.axhline(y=rband_lines[0]-rband_lines[1], linestyle='--',color='blue')
    print 'r done'


    uband_lines=meshgrid(10., 30., 0., 1., .1, .001, 'u', obj)
    plt.axhline(y=uband_lines[0],color='green')
    plt.axhline(y=uband_lines[0]+uband_lines[1], linestyle='--',color='green')
    plt.axhline(y=uband_lines[0]-uband_lines[1], linestyle='--',color='green')
    print 'u done'

    gband_lines=meshgrid(10., 30., 0., 1., .1, .001, 'g', obj)
    plt.axhline(y=gband_lines[0],color='red')
    plt.axhline(y=gband_lines[0]+gband_lines[1], linestyle='--',color='red')
    plt.axhline(y=gband_lines[0]-gband_lines[1], linestyle='--',color='red')
    print 'g done'

    iband_lines=meshgrid(10., 30., 0., 1., .1, .001, 'i', obj)
    plt.axhline(y=iband_lines[0],color='cyan')
    plt.axhline(y=iband_lines[0]+iband_lines[1], linestyle='--',color='cyan')
    plt.axhline(y=iband_lines[0]-iband_lines[1], linestyle='--', color='cyan')
    print 'i done'

    zband_lines=meshgrid(10., 30., 0., 1., .1, .001, 'z', obj)
    plt.axhline(y=zband_lines[0],color='purple')
    plt.axhline(y=zband_lines[0]+zband_lines[1], linestyle='--',color='purple')
    plt.axhline(y=zband_lines[0]-zband_lines[1], linestyle='--',color='purple')
    print 'z done'


    plt.legend(prop={'size':8})
    plt.xlabel('time(days)')
    plt.ylabel('mags')
    plt.title(r'mags for obj %s w/ mean mags & 1$\sigma$ lines'%(obj))
    plt.savefig('data_means_%s.pdf'%(obj))
    os.system('cp data_means_%s.pdf ~/public_html/QSO'%(obj))
