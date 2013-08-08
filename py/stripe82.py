import numpy as np
import pyfits
import utils

class QuasarData:
    """
    Subclass this class with specific data classes
    """
    def get_mags(self, bandname=None):
        """
        Returns all mags if no argument given,
        else if given a bandname it returns only that band
        """
        if bandname:
            mask = [self.bandnames==bandname]
            return self.mags[mask]
        return self.mags

    def get_sigmas(self, bandname=None):
        """
        Returns all sigmas if no argument given,
        else if given a bandname it returns only that band
        """
        if bandname:
            mask = [self.bandnames==bandname]
            return self.sigmas[mask]
        return self.sigmas

    def get_times(self, bandname=None):
        """
        Returns all times if no argument given,
        else if given a bandname it returns only that band
        """
        if bandname:
            mask = [self.bandnames==bandname]
            return self.times[mask]
        return self.times

    def get_bands(self, bandname=None):
        """
        Returns all bands if no argument given,
        else if given a bandname it returns only that band
        """
        if bandname:
            mask = [self.bandnames==bandname]
            return self.bands[mask]
        return self.bands

    def get_bandnames(self, bandname=None):
        """
        Returns all bandnames if no argument given,
        else if given a bandname it returns only that band
        """
        if bandname:
            mask = [self.bandnames==bandname]
            return self.bandnames[mask]
        return self.bandnames

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

    def get_data(self, bandname=None):
        """
        Returns all data in the form of: mags, times, bands, sigmas
        If given a bandname only returns that band
        """
        return [self.get_mags(bandname), self.get_times(bandname),
                self.get_bands(bandname), self.get_sigmas(bandname)]

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
        mask = data['headobjid']==objid
        table = data[mask]
        self.bandlist=['u','g','r','i','z']
        self.banddict={'u':0,'g':1,'r':2,'i':3,'z':4}
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
            self.bands.extend([i]*len(mag))
            self.bandnames.extend([name]*len(mag))

        self.mags = np.array(self.mags)
        self.bands = np.array(self.bands)
        self.bandnames = np.array(self.bandnames)
        self.sigmas = np.array(self.sigmas)
        self.times = np.array(self.times)
        temp.close()


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
            Resamples the data
            Inputs: timestep
            """
            mags, times, sigmas, bands = utils.mock_panstarrs(self.data, timestep)
            bandnames = [self.bandlist[x] for x in bands] 
            self.mags = np.array(mags)
            self.bands = np.array(bands)
            self.bandnames = np.array(bandnames)
            self.sigmas = np.array(sigmas)
            self.times = np.array(times)
