import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import george
import data
import scipy.optimize as op

f = open('newtargetlist.txt', 'r')

obj = int(f.next())

quasar_data = data.Stripe82(obj)

new_data = quasar_data.get_data(bandname='r')

qkernel = .68*george.kernels.ExpKernel(1/50.)

mags = new_data[0]
times = np.log(new_data[1])
mag_errors = new_data[3]


gp = george.GP(qkernel, mean=np.mean(mags))

print qkernel.pars


def nll(p):
    gp.kernel.pars = np.exp(p)
    ll = gp.lnlikelihood(mags)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(p):
    gp.kernel.pars = np.exp(p)
    return -gp.grad_lnlikelihood(mags)

gp.compute(times, yerr = mag_errors)

print gp.lnlikelihood(mags)

p0 = gp.kernel.vector
results = op.minimize(nll, p0, jac=grad_nll)
print np.exp(results.x)
gp.kernel[:] = results.x
gp.compute(times, yerr = mag_errors)
print gp.lnlikelihood(mags)


plt.errorbar(times,mags,yerr=mag_errors,fmt='+')

time_grid = np.linspace(np.min(times), np.max(times), 100)
predict_mean, cov = gp.predict(mags, time_grid)
print np.diag(cov)

print time_grid
print predict_mean

plt.plot(time_grid, predict_mean, '-')

for i in range(5):
    plt.plot(time_grid, gp.sample_conditional(mags, time_grid), alpha=.1)

plt.savefig('gtest.png')

