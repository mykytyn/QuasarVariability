import numpy as np
from QuasarVariability import QuasarVariability
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a = .75
tau = 200.
mean = 19.5

testQ = QuasarVariability(a,tau,mean)

dt = 3.
timegrid = np.arange(0.5*dt,2000,dt)
plt.clf()
pmean,Vpp=testQ.get_mean_vector(timegrid),testQ.get_variance_tensor(timegrid)
pmean=np.array(pmean).reshape(timegrid.shape)
psig=np.sqrt(np.diag(np.array(Vpp)))
plt.plot(timegrid,pmean,'k-')
plt.plot(timegrid,pmean-psig,'k-')
plt.plot(timegrid,pmean+psig,'k-')
for i in range(8):
    maggrid = testQ.get_prior_sample(timegrid)
    maggrid = np.array(maggrid).reshape(timegrid.shape)
    plt.plot(timegrid,maggrid,'k-',alpha=0.25)
plt.savefig("test_prior.png")


testTimes = np.array([500.,1500.,1700.])
testMags = np.array([20.02,19.31,19.75])
testSigmas = np.array([.05,.07,.05])


plt.clf()
plt.errorbar(testTimes,testMags,yerr=testSigmas,linestyle='none',color='black',marker='.')
pmean,Vpp=testQ.get_conditional_mean_and_variance(timegrid,testMags,testTimes,testSigmas)
pmean=np.array(pmean).reshape(timegrid.shape)
psig=np.sqrt(np.diag(np.array(Vpp)))
plt.plot(timegrid,pmean,'k-')
plt.plot(timegrid,pmean-psig,'k-')
plt.plot(timegrid,pmean+psig,'k-')
for i in range(8):
    maggrid = testQ.get_conditional_sample(timegrid,testMags,testTimes,testSigmas)
    maggrid = np.array(maggrid).reshape(timegrid.shape)
    plt.plot(timegrid,maggrid,'k-',alpha=0.25)
plt.savefig("test_posterior.png")
