#/usr/bin/env python
import gdal
from glob import glob

class read_l8(object):
    '''
    read in the l8 datasets, toa and angles.
    '''
    def __init__(self,
                 toa_dir,
                 tile,
                 year,
                 month,
                 day,
                 bands = None
                ):
        self.toa_dir   = toa_dir
        self.tile      = tile
        self.year      = year
        self.month     = month
        self.day       = day
        if bands is None:
            self.bands = np.arange(1, 8)
        else:
            self.bands = bands

    def _get_toa(self,):
        composite    = glob(self.toa_dir + '/LC08_L1TP_%03d%03d_%04d%02d%02d_*_01_T1_toa_band1.tif' \
                         % ( self.tile[0], self.tile[1], self.year, self.month, self.day))[0].split('/')[-1].split('_')[:-2]
        composite[4] = '*'
        header       = '_'.join(composite)    
        
