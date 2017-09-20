#/usr/bin/env python
import gdal
import os

class get_s2_toa(object):
    '''
    A class reading S2 toa reflectance, taken the directory, date and bands needed,
    It will read in the cloud mask as well, if no cloud.tiff, then call the classification
    algorithm to get the cloud mask and save it.
    '''
    def __init__(self, 
                 s2_toa_dir,
                 s2_tile, 
                 year, month, day,
                 bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A'):
        self.s2_toa_dir = s2_toa_dir
        self.s2_tile    = s2_tile
        self.year       = year
        self.month      = month
        self.day        = day
        self.bands      = bands # selected bands
        self.s2_bands   = 'B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B8A' #all bands

    def get_s2_toa(self,):

        self.s2_file_dir = os.path.join(self.s2_toa_dir, self.s2_tile[:-3],\
                                        self.s2_tile[-3], self.s2_tile[-2:],\
                                        str(self.year), str(self.month), str(self.day))
        # open the created vrt file with 10 meter, 20 meter and 60 meter 
        # grouped togehter and use gdal memory map to open it
        g = gdal.Open('/'.join(self.s2_file_dir+'10meter.vrt'))
        data= g.GetVirtualMemArray()
        b2,b3,b4,b8 = data
        g1 = gdal.Open('/'.join(self.s2_dir+'20meter.vrt'))
        data1 = g1.GetVirtualMemArray()
        b5, b6, b7, b8a, b11, b12 = data1
        g2 = gdal.Open('/'.join(self.s2_dir+'60meter.vrt'))
        data2 = g2.GetVirtualMemArray()
        b1, b9, b10 = data2
        img = dict(zip(self.s2_bands, [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b8a]))
        if self.bands is not None:
            selected = {k: img[k] for k in self.bands}
            return selected
        else:
            return img
