import numpy as np
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rmag_utils import *
import os


plt.figure()

fn=open('ids.txt','r')
objs=fn.readlines()
objs=[o.strip('\n') for o in objs]
objs=[int(o) for o in objs]
objs=objs[0:1]

for obj in objs:
    plt.clf()
    plt.subplots_adjust(hspace=0,top=0.95)
    matplotlib.rc('xtick',labelsize=8)
    matplotlib.rc('ytick',labelsize=8)

    ax1=plt.subplot(511)
    plot_mag('u',obj)
    missing_points('u',obj)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(512)
    plot_mag('g',obj)
    missing_points('g',obj)
    plt.setp(ax2.get_xticklabels(),visible=False)    

    ax3=plt.subplot(513)
    plot_mag('r',obj)
    missing_points('r',obj)
    plt.setp(ax3.get_xticklabels(),visible=False)    

    ax4=plt.subplot(514)
    plot_mag('i',obj)
    missing_points('i',obj)
    plt.setp(ax4.get_xticklabels(),visible=False)    

    plt.subplot(515)
    missing_points('z',obj)
    plot_mag('z',obj)

    plt.xlabel('time(days)')    
    plt.suptitle(r'obj %s'%(obj))
    plt.savefig('allmags.pdf')
    os.system('cp allmags.pdf ~/public_html/QSO')
