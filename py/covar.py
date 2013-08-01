import numpy as np
from lnprob import ln_prob
import pyfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




ts = np.arange(0,1000.,50.)

tau = 200.

tt,tt = np.meshgrid(ts,ts)
ti,tj = tt.shape

V = np.zeros_like(tt)

for i in range(ti):
    for j in range(tj):
        V[i][j] = np.exp(-np.abs(ts[i]-ts[j])/tau)


mean = np.ones_like(ts)
#mean = [x for x in range(20)]

samples = np.random.multivariate_normal(mean,V,5)
print samples

for i in range(len(samples)):
    plt.plot(ts,samples[i],'.-')
plt.savefig('samples.png')
