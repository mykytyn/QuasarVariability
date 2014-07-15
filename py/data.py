import numpy as np
import pyfits
import utils


class QuasarData:
    """
    Subclass this class with specific data classes
    """

    def get_mags(self, bandname=None, include_bad=False):
        """
        Returns all mags if no argument given,
        else if given a bandname it returns only that band
        """
        mask = np.ones(len(self.mags),dtype=bool)
        if bandname:
            mask = np.logical_and(mask, self.bandnames == bandname)
        if not include_bad:
            mask = np.logical_and(mask, np.logical_not(self.bad))
        return self.mags[mask]

    def get_sigmas(self, bandname=None, include_bad=False):
        """
        Returns all sigmas if no argument given,
        else if given a bandname it returns only that band
        """
        mask = np.ones(len(self.mags),dtype=bool)
        if bandname:
            mask = np.logical_and(mask, self.bandnames == bandname)
        if not include_bad:
            mask = np.logical_and(mask, np.logical_not(self.bad))
        return self.sigmas[mask]

    def get_times(self, bandname=None, include_bad=False):
        """
        Returns all times if no argument given,
        else if given a bandname it returns only that band
        """
        mask = np.ones(len(self.mags),dtype=bool)
        if bandname:
            mask = np.logical_and(mask, self.bandnames == bandname)
        if not include_bad:
            mask = np.logical_and(mask, np.logical_not(self.bad))
        return self.times[mask]

    def get_bands(self, bandname=None, include_bad=False):
        """
        Returns all bands if no argument given,
        else if given a bandname it returns only that band
        """
        mask = np.ones(len(self.mags),dtype=bool)
        if bandname:
            mask = np.logical_and(mask, self.bandnames == bandname)
        if not include_bad:
            mask = np.logical_and(mask, np.logical_not(self.bad))
        return self.bands[mask]

    def get_bandnames(self, bandname=None, include_bad=False):
        """
        Returns all bandnames if no argument given,
        else if given a bandname it returns only that band
        """
        mask = np.ones(len(self.mags),dtype=bool)
        if bandname:
            mask = np.logical_and(mask, self.bandnames == bandname)
        if not include_bad:
            mask = np.logical_and(mask, np.logical_not(self.bad))
        return self.bandnames[mask]

    def get_bandlist(self):
        """
        Returns a list of the string bandnames,
        ordered correctly
        """
        return self.bandlist

    def get_banddict(self):
        """
        Returns a dict allowing conversion from string bandname
        into the corresponding number
        """
        return self.banddict

    def get_bad_mask(self, bandname=None):
        if bandname:
            mask = [self.bandnames==bandname]
            return self.bad[mask]
        return self.bad

    def get_data(self, bandname=None, include_bad=False):
        """
        Returns all data in the form of: mags, times, bands, sigmas
        If given a bandname only returns that band
        """
        return [self.get_mags(bandname, include_bad), self.get_times(bandname, include_bad), self.get_bands(bandname, include_bad), self.get_sigmas(bandname, include_bad), self.get_bad_mask(bandname)]

class Stripe82 (QuasarData):
    """
    This class is used for stripe82 data
    """
    def __init__(self, objid, fn='quasar'):
        """
        Inputs: objid we care about
                fn (optional) name of s82 file (default is what I call it)
        """
        temp = pyfits.open('{}.fits'.format(fn))
        data = temp[1].data
        mask = data['headobjid'] == objid
        table = data[mask]
        self.bandlist=['u', 'g', 'r', 'i', 'z']
        self.banddict={'u':0, 'g':1, 'r':2, 'i':3, 'z':4}
        self.mags = []
        self.bands = []
        self.bandnames = []
        self.sigmas = []
        self.times = []

        for i in range(5):
            name = self.bandlist[i]
            mag = table['psfMag_{}'.format(name)]
            self.mags.extend(mag)
            self.sigmas.extend(table['psfMagErr_{}'.format(name)])
            self.times.extend(table['mjd_{}'.format(name)])
            self.bands.extend([i] * len(mag))
            self.bandnames.extend([name] * len(mag))

        self.mags = np.array(self.mags)
        self.bands = np.array(self.bands)
        self.bandnames = np.array(self.bandnames)
        self.sigmas = np.array(self.sigmas)
        self.times = np.array(self.times)
        self.bad = np.zeros(len(self.times), dtype=bool)
        temp.close()

    def remove_bad_data(self, cutoff, interval):
        """
        Inputs: cutoff: cutoff in mags
        interval: interval in days
        This finds points which have a variance greater than cutoff, and flags points that are closer than interval to those points
        """
        
        mask = self.sigmas > cutoff
        badtimes = [x for x in self.times if np.any(
                np.abs(x - self.times[mask]) < interval)]
        mask2 = np.zeros(len(self.times), dtype=bool)
        for x in badtimes:
            mask2 = np.logical_or(mask2, self.times == x)
        self.bad = mask2
        mask2 = np.logical_not(mask2)

        


class MockPanstarrs(Stripe82):
        def __init__(self, objid, timestep=10., fn='quasar'):
            """
            Inputs: objid we care about
            fn (optional) name of s82 file (default is what I call it)
            """
            self.data = Stripe82(objid,fn)
            self.bandlist=['u', 'g', 'r', 'i', 'z']
            self.banddict={'u':0, 'g':1, 'r':2, 'i':3, 'z':4}
            mags, times, sigmas, bands = utils.mock_panstarrs(self.data, timestep)
            bandnames = [self.bandlist[x] for x in bands] 
            self.mags = np.array(mags)
            self.bands = np.array(bands)
            self.bandnames = np.array(bandnames)
            self.sigmas = np.array(sigmas)
            self.times = np.array(times)


        def resample(self, timestep=10.):
            """
            Resamples the data to simulate panstarrs
            Inputs: timestep
            """
            mags, times, sigmas, bands = utils.mock_panstarrs(self.data, timestep)
            bandnames = [self.bandlist[x] for x in bands] 
            self.mags = np.array(mags)
            self.bands = np.array(bands)
            self.bandnames = np.array(bandnames)
            self.sigmas = np.array(sigmas)
            self.times = np.array(times)
