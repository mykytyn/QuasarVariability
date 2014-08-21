import numpy as np
import pyfits
import random

table = pyfits.open('quasar.fits')
data = table[1].data
print len(set(data['headobjid']))
print len(data['headobjid'])
assert False
random_sample =  random.sample(data['headobjid'],256)

f = open('256sample.txt', 'w')
for samp in random_sample:
    f.write('{}\n'.format(samp))

f.close()

