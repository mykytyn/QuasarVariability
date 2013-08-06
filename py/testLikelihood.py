import numpy as np
from QuasarVariability import QuasarVariability,CovarianceFunction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mag_utils import meshgrid
import pyfits

bands_dict = {0:'u',1:'g',2:'r',3:'i',4:'z'}

def make_prior_plots(testQ,timegrid,bands,pmean,psig,means,num_samps=8):
    maggrids = []
    plt.subplots_adjust(hspace=0,top=.95)
    matplotlib.rc('xtick',labelsize=6)
    matplotlib.rc('ytick',labelsize=6)
    for i in range(num_samps):
        maggrid = testQ.get_prior_sample(timegrid,bands)
        maggrid = np.array(maggrid).reshape(timegrid.shape)
        maggrids.append(maggrid)
        
    for i in range(5):
        ax=plt.subplot(511+i)
        mask = [bands==i]

        btimegrid = timegrid[mask]
        bpmean = pmean[mask]
        bpsig = psig[mask]
        plt.plot(btimegrid,bpmean,'k-')
        plt.plot(btimegrid,bpmean-bpsig,'k-')
        plt.plot(btimegrid,bpmean+bpsig,'k-')
        for j in range(num_samps):
            maggrid = maggrids[j]
            plt.plot(btimegrid,maggrid[mask],'k-',alpha=0.25)
        plt.setp(ax.get_xticklabels(),visible=(i==4))
        plt.ylabel('%s' % bands_dict[i])
        plt.ylim(means[i]-.75,means[i]+.75)

def make_posterior_plots(testQ,test_times,test_mags,test_bands,test_sigmas,timegrid,bands,pmean,psig,means,num_samps=8):
    maggrids = []
    plt.subplots_adjust(hspace=0,top=.95)
    matplotlib.rc('xtick',labelsize=8)
    matplotlib.rc('ytick',labelsize=8)

    for i in range(num_samps):
        maggrid = testQ.get_conditional_sample(timegrid, bands, test_mags, test_times,
                                               test_bands,test_sigmas)
        maggrid = np.array(maggrid).reshape(timegrid.shape)
        maggrids.append(maggrid)

    for i in range(5):
        ax=plt.subplot(511+i)
        mask = [bands==i]
        test_mask = [test_bands==i]
        plt.ylabel('%s' % bands_dict[i])
        plt.errorbar(test_times[test_mask],test_mags[test_mask],yerr=test_sigmas[test_mask],linestyle='none',color='black',marker='.')
        btimegrid = timegrid[mask]
        bpmean = pmean[mask]
        bpsig = psig[mask]
        plt.plot(btimegrid,bpmean,'k-')
        plt.plot(btimegrid,bpmean-bpsig,'k-')
        plt.plot(btimegrid,bpmean+bpsig,'k-')
        for j in range(num_samps):
            maggrid=maggrids[j]
            plt.plot(btimegrid,maggrid[mask],'k-',alpha=0.25)
        plt.setp(ax.get_xticklabels(),visible=(i==4))
        plt.ylim(means[i]-.75,means[i]+.75)

tau = 200.

bandnames = ['u','g','r','i','z']
obj = 588015509825912905
dt = 10.
fulltimegrid = []
fullbands = []

for i in range(5):
    timegrid = np.arange(0.5*dt+51500,54500,dt)
    bands = [i]*len(timegrid)
    fulltimegrid.extend(timegrid)
    fullbands.extend(bands)

timegrid = np.array(fulltimegrid)
bands = np.array(fullbands)

means = []
amps = []
for name in bandnames:
    mean, a = meshgrid(10., 30., 0., 1., 0.3, 0.008,name,obj)
    means.append(mean)
    amps.append(a)

print means
print amps
means = np.array(means)
amps = np.array(amps)
plt.clf()
testQ = QuasarVariability(CovarianceFunction(amps,tau),means)
pmean=testQ.get_mean_vector(timegrid,bands)
Vpp =testQ.get_variance_tensor(timegrid,bands)
pmean=np.array(pmean).reshape(timegrid.shape)
psig=np.sqrt(np.diag(np.array(Vpp)))
make_prior_plots(testQ,timegrid,bands,pmean,psig,means)
plt.savefig("test_prior2.png")

hdulist = pyfits.open('quasar.fits')
table = hdulist[1].data
mask = table['headobjid']==obj
table = table[mask]

testMags = []
testSigmas = []
testTimes = []
testBands = []

for i in range(5):
    name = bands_dict[i]
    mag = table['psfMag_%s' % name]
    err = table['psfMagerr_%s' % name]
    time = table['mjd_%s' % name]
    band = [i]*len(mag)
    testMags.extend(mag)
    testSigmas.extend(err)
    testBands.extend(band)
    testTimes.extend(time)

testMags = np.array(testMags)
testSigmas = np.array(testSigmas)
testBands = np.array(testBands)
testTimes = np.array(testTimes)

plt.clf()

pmean,Vpp=testQ.get_conditional_mean_and_variance(timegrid, bands, testMags, testTimes,
                                                  testBands, testSigmas)
pmean=np.array(pmean).reshape(timegrid.shape)
psig=np.sqrt(np.diag(np.array(Vpp)))
make_posterior_plots(testQ,testTimes,testMags,testBands,testSigmas,timegrid,bands,pmean,psig,means)
plt.savefig("test_posterior2.png")
