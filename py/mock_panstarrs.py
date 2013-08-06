import numpy as np
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mag_utils import *
import os
from QuasarVariability import QuasarVariability

fn=open('ids.txt','r')
objs=fn.readlines()
objs=[o.strip('\n') for o in objs]
objs=[int(o) for o in objs]
objs=objs[0:1]

plt.figure()

for obj in objs:
    data=mock_panstarrs(obj,10)
    plt.clf()
    plt.subplots_adjust(hspace=0,top=0.95)
    matplotlib.rc('xtick',labelsize=8)
    matplotlib.rc('ytick',labelsize=8)

    ax1=plt.subplot(511)
    plt.setp(ax1.get_xticklabels(),visible=False)
    ax1.set_ylabel('u')
    ax1.set_xlim(51000,54500)

    ax2=plt.subplot(512)
    plt.setp(ax2.get_xticklabels(),visible=False)
    ax2.set_ylabel('g')
    ax2.set_xlim(51000,54500)

    ax3=plt.subplot(513)
    plt.setp(ax3.get_xticklabels(),visible=False)    
    ax3.set_ylabel('r')
    ax3.set_xlim(51000,54500)

    ax4=plt.subplot(514)
    plt.setp(ax4.get_xticklabels(),visible=False)    
    ax4.set_ylabel('i')
    ax4.set_xlim(51000,54500)

    ax5=plt.subplot(515)
    ax5.set_ylabel('z')
    ax5.set_xlim(51000,54500)

    print data
    umags=[]
    gmags=[]
    rmags=[]
    imags=[]
    zmags=[]

    for key,value in data.iteritems():
        if value=='u':
            umags.append(key[1])
            print key[1]
            ax1.errorbar(key[0],key[1],yerr=key[2],marker='.',color='black',ls='none')
        
        if value=='g':
            gmags.append(key[1])
            print key[1]
            ax2.errorbar(key[0],key[1],yerr=key[2],marker='.',color='black',ls='none')

        if value=='r':
            rmags.append(key[1])
            print key[1]
            ax3.errorbar(key[0],key[1],yerr=key[2],marker='.',color='black',ls='none')

        if value=='i':
            imags.append(key[1])
            print key[1]
            ax4.errorbar(key[0],key[1],yerr=key[2],marker='.',color='black',ls='none')
    
        if value=='z':
            zmags.append(key[1])
            print key[1]
            ax5.errorbar(key[0],key[1],yerr=key[2],marker='.',color='black',ls='none')

    ax1.set_ylim(np.mean(umags)-0.75,np.mean(umags)+0.75)
    ax2.set_ylim(np.mean(gmags)-0.75,np.mean(gmags)+0.75)
    ax3.set_ylim(np.mean(rmags)-0.75,np.mean(rmags)+0.75)
    ax4.set_ylim(np.mean(imags)-0.75,np.mean(imags)+0.75)
    ax5.set_ylim(np.mean(zmags)-0.75,np.mean(zmags)+0.75)
    
    plt.xlabel('time(days)')    
    plt.suptitle(r'obj %s'%(obj))
    plt.savefig('mock_panstarrs_test.pdf')
    os.system('cp mock_panstarrs_test.pdf ~/public_html/QSO')


