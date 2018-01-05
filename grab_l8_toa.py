#/usr/bin/env python
import os
import sys
sys.path.insert(0, 'python')
import gdal
import numpy as np
from glob import glob
import subprocess
from datetime import datetime
from multi_process import parmap

def gdal_reader(fname):
    g = gdal.Open(fname)
    if g is None:
        raise IOError
    else:
        return g.ReadAsArray()

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
                 bands = None,
                 angle_exe = '/home/ucfafyi/DATA/S2_MODIS/l_data/l8_angles/l8_angles'
                ):
        self.toa_dir   = toa_dir
        self.tile      = tile
        self.year      = year
        self.month     = month
        self.day       = day
        if bands is None:
            self.bands = np.arange(1, 8)
        else:
            self.bands = np.array(bands)
        self.angle_exe = angle_exe
        composite      = glob(self.toa_dir + '/LC08_L1TP_%03d%03d_%04d%02d%02d_*_01_??_toa_band1.tif' \
                         % ( self.tile[0], self.tile[1], self.year, self.month, self.day))[0].split('/')[-1].split('_')[:-2]
        self.header    = '_'.join(composite)    
        self.toa_file  = [self.toa_dir + '/%s_b%d.tif'%(self.header, i) for i in self.bands]
        self.mete_file =  self.toa_dir + '/%s_MTL.txt'%self.header 
        self.qa_file   =  self.toa_dir + '/%s_bqa.tif'%self.header
        try:
            self.saa_sza = [glob(self.toa_dir + '/%s_solar_B%02d.img' %(self.header, i))[0] for i in self.bands]
            self.vaa_vza = [glob(self.toa_dir + '/%s_sensor_B%02d.img'%(self.header, i))[0] for i in self.bands]
        except:
            ang_file     = self.toa_dir + '/%s_ANG.txt'%self.header
            cwd = os.getcwd()
            os.chdir(self.toa_dir)
            f            =  lambda band: subprocess.call([self.angle_exe, ang_file, \
                                                          'BOTH', '1', '-f', '-32768', '-b', str(band)])
            parmap(f, self.bands)
            os.chdir(cwd)
            self.saa_sza = [self.toa_dir + '/%s_solar_B%02d.img' %(self.header, i) for i in self.bands]
            self.vaa_vza = [self.toa_dir + '/%s_sensor_B%02d.img'%(self.header, i) for i in self.bands]
        try:
            scale, offset = self._get_scale()
        except:
            raise IOError, 'Failed read in scalling factors.'

    def _get_toa(self,):
        try:
            scale, offset = self._get_scale()
        except:
            raise IOError, 'Failed read in scalling factors.'
        bands_scale  = scale [self.bands-1]
        bands_offset = offset[self.bands-1] 
        toa          = np.array(parmap(gdal_reader, self.toa_file)).astype(float) * \
                                bands_scale[...,None, None] + bands_offset[...,None, None]
        qa_mask  = self._get_qa()
        sza      = self._get_angles()[1]
        toa      = toa / np.cos(np.deg2rad(sza))
        toa_mask = toa < 0
        mask     = qa_mask | toa_mask | sza.mask
        toa      = np.ma.array(toa, mask=mask)
        return toa

    def _get_angles(self,):
        saa, sza = np.array(parmap(gdal_reader, self.saa_sza)).astype(float).transpose(1,0,2,3)/100.
        vaa, vza = np.array(parmap(gdal_reader, self.vaa_vza)).astype(float).transpose(1,0,2,3)/100.
        saa = np.ma.array(saa, mask = ((saa > 180) | (saa < -180)))
        sza = np.ma.array(sza, mask = ((sza > 90 ) | (sza < 0   )))
        vaa = np.ma.array(vaa, mask = ((vaa > 180) | (vaa < -180)))
        vza = np.ma.array(vza, mask = ((vza > 90 ) | (vza < 0   )))
        saa.mask = sza.mask = vaa.mask = vza.mask = (saa.mask | sza.mask | vaa.mask | vza.mask)
        return saa, sza, vaa, vza

    def _get_scale(self,):
        scale, offset = [], []
        with open( self.mete_file, 'rb') as f:
            for line in f:
                if 'REFLECTANCE_MULT_BAND' in line:
                    scale.append(float(line.split()[-1]))
                elif 'REFLECTANCE_ADD_BAND' in line:
                    offset.append(float(line.split()[-1]))
                elif 'DATE_ACQUIRED' in line:
                    date = line.split()[-1]
                elif 'SCENE_CENTER_TIME' in line:
                    time = line.split()[-1]
        datetime_str  = date + time
        self.sen_time = datetime.strptime(datetime_str.split('.')[0], '%Y-%m-%d"%H:%M:%S')
        return np.array(scale), np.array(offset)

    def _get_qa(self,):
        bqa = gdal_reader(self.qa_file)
        qa_mask = ~((bqa >= 2720) & (bqa <= 2732))
        return qa_mask
if __name__ == '__main__':
    l8 = read_l8('/home/ucfafyi/DATA/S2_MODIS/l_data/', (123, 34), 2017, 4, 21, bands=[2,3,4,5,6,7])
    #toa = l8._get_toa()
