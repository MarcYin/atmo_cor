#/usr/bin/env python
import os
import sys
sys.path.insert(0,'python')
import gdal
import numpy as np
from numpy import clip, uint8
from glob import glob
import logging
from Py6S import *
import cPickle as pkl
from multi_process import parmap
from grab_l8_toa import read_l8
from reproject import reproject_data
import warnings
warnings.filterwarnings("ignore")

class atmospheric_correction(object):
    '''
    A class doing the atmospheric coprrection with the input of TOA reflectance
    angles, elevation and emulators of 6S from TOA to surface reflectance.
    '''
    def __init__(self,
                 year, 
                 month, 
                 day,
                 l8_tile,
                 l8_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/l_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 emus_dir    = '/home/ucfafyi/DATA/Multiply/emus/',
                 ):              
        
        self.year        = year
        self.month       = month
        self.day         = day
        self.l8_tile     = l8_tile
        self.l8_toa_dir  = l8_toa_dir
        self.global_dem  = global_dem
        self.emus_dir    = emus_dir
        self.sur_refs     = {}
        self.bands       = [1, 2, 3, 4, 5, 6, 7]
	self.logger = logging.getLogger('Landsat 8 Atmospheric Correction')
	self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:       
	    ch = logging.StreamHandler()
	    ch.setLevel(logging.DEBUG)
	    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	    ch.setFormatter(formatter)
	    self.logger.addHandler(ch)

    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap.pkl'%(self.sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp.pkl'%(self.sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp.pkl'%(self.sensor))[0]
        f = lambda em: pkl.load(open(em, 'rb'))
        self.xap_emus, self.xbp_emus, self.xcp_emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def atmospheric_correction(self,):

        self.logger.propagate = False
        self.sensor = 'OLI'
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        l8   = read_l8(self.l8_toa_dir, self.l8_tile, self.year, self.month, self.day, bands = self.bands)
        self.l8_header = l8.header
        self.example_file = self.l8_toa_dir + '/%s_b%d.tif'%(l8.header, 1)
        self.logger.info('Reading in the reflectance.')
        self.toa = l8._get_toa()
        self.logger.info('Reading in the angles')
        self.saa, self.sza, self.vaa, self.vza = l8._get_angles()
        self.saa[self.saa.mask] = self.sza[self.sza.mask] = \
        self.vaa[self.vaa.mask] = self.vza[self.vza.mask] = np.nan
        self.aot, self.tcwv, self.tco3, self.ele = self._get_control_variables()
        self.shape = self.toa.shape[1:3]
        self._block_size = 3000
        self._num_blocks_x, self._num_blocks_y = int(np.ceil(1. * self.shape[0] / self._block_size)), int(np.ceil(1. * self.shape[1] / self._block_size))
        self._mean_size  = 60
        rows              = np.repeat(np.arange(self._num_blocks_x), self._num_blocks_y)
        columns           = np.tile  (np.arange(self._num_blocks_y), self._num_blocks_x)
        blocks            = zip(rows, columns)
        self.logger.info('Doing correction')
        ret = parmap(self._block_correction_emus_xa_xb_xc, blocks)
        self.boa = np.array([i[2] for i in ret]).reshape(self._num_blocks_x, self._num_blocks_y, self.toa.shape[0], \
                             self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(self.toa.shape[0], \
                             self._num_blocks_x*self._block_size, self._num_blocks_y*self._block_size)[:, : self.shape[0], : self.shape[1]]
        self.boa_rgb = np.clip(self.boa[[3,2,1]].transpose(1,2,0) * 255 / 0.255, 0, 255).astype(uint8)
        self.toa_rgb = np.clip(self.toa[[3,2,1]].transpose(1,2,0) * 255 / 0.255, 0, 255).astype(uint8)
        self.logger.info('Saving corrected results')
        self._save_rgb(self.toa_rgb, 'TOA_RGB', self.example_file)
        self._save_rgb(self.boa_rgb, 'BOA_RGB', self.example_file)
        self._save_img(self.boa, self.bands)

 
    def _get_control_variables(self,):

        aot  = reproject_data(self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, 'aot'), \
                              self.example_file, outputType= gdal.GDT_Float32).data

        tcwv = reproject_data(self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, 'tcwv'), \
                              self.example_file, outputType= gdal.GDT_Float32).data

        tco3 = reproject_data(self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, 'tco3'), \
                              self.example_file, outputType= gdal.GDT_Float32).data
        ele = reproject_data(self.global_dem, self.example_file, outputType= gdal.GDT_Float32).data
        mask = ~np.isfinite(ele)
        ele[mask] = np.interp(np.flatnonzero(mask), \
                              np.flatnonzero(~mask), ele[~mask]) # simple interpolation

        return aot, tcwv, tco3, ele

    def _block_helper(self, val, block):
        i, j      = block
        slice_x   = slice(i*self._block_size,(i+1)*self._block_size, 1)
        slice_y   = slice(j*self._block_size,(j+1)*self._block_size, 1)
        if   val.ndim == 2:
            temp    = np.zeros((self._block_size, self._block_size))
            temp[:] = np.nan
            temp  [ : min((i+1) * self._block_size, self.shape[0]) - i * self._block_size, \
                    : min((j+1) * self._block_size, self.shape[1]) - j * self._block_size] = val[slice_x,slice_y]
        elif val.ndim == 3:
            temp    = np.zeros((val.shape[0], self._block_size, self._block_size))
            temp[:] = np.nan
            temp  [:, : min((i+1) * self._block_size, self.shape[0]) - i * self._block_size, \
                      : min((j+1) * self._block_size, self.shape[1]) - j * self._block_size] = val[:, slice_x,slice_y]
        return temp

    def _save_rgb(self, rgb_array, name, source_image):
        g            = gdal.Open(source_image)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        nx, ny = rgb_array.shape[:2]
        outputFileName = self.l8_toa_dir + '/%s_%s.tif'%(self.l8_header, name)
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 3, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).WriteArray(rgb_array[:,:,0])
        dst_ds.GetRasterBand(2).WriteArray(rgb_array[:,:,1])
        dst_ds.GetRasterBand(3).WriteArray(rgb_array[:,:,2])
        dst_ds.FlushCache()
        dst_ds = None

    def _save_img(self, refs, bands):
        g            = gdal.Open(self.example_file)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        bands_refs   = zip(bands, refs)
        f            = lambda band_ref: self._save_band(band_ref, projection = projection, geotransform = geotransform)
        parmap(f, bands_refs)
 
    def _save_band(self, band_ref, projection, geotransform):
        band, ref = band_ref
        nx, ny = ref.shape
        outputFileName = self.l8_toa_dir + '/%s_%s_sur.tif'%(self.l8_header, band)
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)    
        dst_ds.SetProjection(projection) 
        dst_ds.GetRasterBand(1).WriteArray(ref)
        dst_ds.FlushCache()                  
        dst_ds = None

    def _block_correction_emus_xa_xb_xc(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i+1,j+1))
        toa       = self._block_helper(self.toa,    block)
        vza       = self._block_helper(self.vza,    block) * np.pi / 180.
        vaa       = self._block_helper(self.vaa,    block) * np.pi / 180.
        sza       = self._block_helper(self.sza[0], block) * np.pi / 180.
        saa       = self._block_helper(self.saa[0], block) * np.pi / 180.
        tcwv      = self._block_helper(self.tcwv,   block)
        tco3      = self._block_helper(self.tco3,   block)
        aot       = self._block_helper(self.aot,    block)
        elevation = self._block_helper(self.ele,    block) / 1000.
        corfs = []
        for bi, band in enumerate(range(len(self.bands))):    
            p = [self._block_mean(va, self._mean_size).ravel() for va in [np.cos(sza), \
                 np.cos(vza[bi]), np.cos(saa - vaa[bi]), aot, tcwv, tco3, elevation]] 
            a = self.xap_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            b = self.xbp_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            c = self.xcp_emus[band].predict(np.array(p).T)[0].reshape(self._block_size//self._mean_size, \
                                                                      self._block_size//self._mean_size)
            a = np.repeat(np.repeat(a, self._mean_size, axis=0), self._mean_size, axis=1)
            b = np.repeat(np.repeat(b, self._mean_size, axis=0), self._mean_size, axis=1)
            c = np.repeat(np.repeat(c, self._mean_size, axis=0), self._mean_size, axis=1)
            y     = a * toa[bi] -b
            corf  = y / (1 + c*y)
            corfs.append(corf)
        boa = np.array(corfs)
        return [i, j, boa]

    def _block_mean(self, data, block_size):
        x_size, y_size = data.shape
        x_blocks       = x_size//block_size
        y_blocks       = y_size//block_size
        data           = data.copy().reshape(x_blocks, block_size, y_blocks, block_size)        
        small_data     = np.nanmean(data, axis=(3,1))
        return small_data
 
if __name__=='__main__':
    atmo_cor = atmospheric_correction(2017, 7, 10, (123, 34),)
    atmo_cor.atmospheric_correction()
