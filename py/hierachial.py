
import cPickle
import numpy as np
from scipy.optimize import leastsq
import matplotlib
matplotlib.use('Agg')
import utils


def main():
    f = open('newtargetlist.txt', 'r')
    first = int(f.readline())
    print first

def make_large_triangle_plot():
    """
    still working on it
    everything is hardcoded
    might be removed soon
    """

    f = open('newtargetlist.txt', 'r')
    hugeflatchains = []
    hugelnprobabilities = []
    for numobj,obj in enumerate(f):
        prefix = obj
        obj = int(obj)
        g = open("{}.pickle".format(obj))
        quasar, quasar_data, flatchain, lnprobability, labels = cPickle.load(g)
        g.close()
        hugeflatchains.append(flatchain)
        hugelnprobabilities.extend(lnprobability)
        print flatchain.shape, lnprobability.shape, len(labels)
    hugeflatchain = np.concatenate(hugeflatchains)
    print hugeflatchain.shape
    hugelnprobability = np.hstack(hugelnprobabilities)
    print hugelnprobability.shape
    utils.make_triangle_plot(hugelnprobability, hugeflatchain, labels).savefig('hugetriangle.png')

if __name__ == '__main__':
    main()
