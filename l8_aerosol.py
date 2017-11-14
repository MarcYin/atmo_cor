#/usr/bin/env python 
import os
import sys
sys.path.insert(0, 'python')
import gdal
import json
import datetime
import logging
import numpy as np
from grab_l8_toa import read_l8
from ddv import ddv
from glob import glob
from scipy import signal, ndimage
import cPickle as pkl
from osgeo import osr
from multi_process import parmap
from reproject import reproject_data
from get_brdf import get_brdf_six
from grab_uncertainty import grab_uncertainty
from atmo_paras_optimization_new import solving_atmo_paras
from psf_optimize import psf_optimize
from spatial_mapping import Find_corresponding_pixels

class solve_aerosol(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,
                 year,
                 month,
                 day,
                 emus_dir    = '/home/ucfafyi/DATA/Multiply/emus/',
                 mcd43_dir   = '/data/selene/ucfajlg/Ujia/MCD43/',
                 l8_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/l_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 cams_dir    = '/home/ucfafyi/DATA/Multiply/cams/',
                 l8_tile     = (123, 34),
                 l8_psf      = None,
                 qa_thresh   = 255,
                 aero_res    = 3000, # resolution for aerosol retrival in meters should be larger than 500
                 ):

        self.year        = year
        self.month       = month
        self.day         = day
        self.date        = datetime.datetime(self.year, self.month, self.day)
        self.doy         = self.date.timetuple().tm_yday
        self.mcd43_dir   = mcd43_dir
        self.emus_dir    = emus_dir
        self.qa_thresh   = qa_thresh
        self.l8_toa_dir  = l8_toa_dir
        self.global_dem  = global_dem
        self.cams_dir    = cams_dir
        self.l8_tile     = l8_tile
        self.l8_psf      = l8_psf
        self.bands       = [2, 3, 4, 5, 6, 7]
        self.boa_bands   = [469, 555, 645, 869, 1640, 2130]
        self.aero_res    = aero_res
        self.mcd43_tmp   = '%s/MCD43A1.A%d%03d.%s.006.*.hdf'
        self.spectral_transform = [[1.0425211806,      1.03763437575,     1.02046102587,     0.999167480738,  1.00072211685,    0.955317665361  ], 
                                   [0.000960797104206, -0.00263498369438, -0.00179952807464, 0.0018999624331, -0.0072213121738, 0.00782954328347]] 
    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap.pkl'%(self.sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp.pkl'%(self.sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp.pkl'%(self.sensor))[0]
        f = lambda em: pkl.load(open(em, 'rb'))
        self.emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def gaussian(self, xstd, ystd, angle, norm = True):
        win = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        winx = int(round(win*(2**0.5)))
        winy = int(round(win*(2**0.5)))
        xgaus = signal.gaussian(winx, xstd)
        ygaus = signal.gaussian(winy, ystd)
        gaus  = np.outer(xgaus, ygaus)
        r_gaus = ndimage.interpolation.rotate(gaus, angle, reshape=True)
        center = np.array(r_gaus.shape)/2
        cgaus = r_gaus[center[0]-win/2: center[0]+win/2, center[1]-win/2:center[1]+win/2]
        if norm:
            return cgaus/cgaus.sum()
        else:
            return cgaus

    def _l8_aerosol(self,):
        self.logger.propagate = False
        self.logger.info('Start to retrieve atmospheric parameters.')
        l8 = read_l8(self.l8_toa_dir, self.l8_tile, self.year, self.month, self.day, bands = self.bands)
        self.logger.info('Reading in TOA reflectance.')
        self.toa      = l8._get_toa()
        self.sza      = l8.sza
        self.vza      = l8.vza
        self.saa      = l8.saa
        self.vaa      = l8.vaa
        self.sen_time = l8.sen_time
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        self.logger.info('Find corresponding pixels between L8 and MODIS tiles')
        self.example_file = self.l8_toa_dir + '/%s_b%d.tif'%(l8.header, 1)
        tiles = Find_corresponding_pixels(self.example_file, destination_res=500)
        if len(tiles.keys())>1:
            self.logger.info('This Landsat 8 tile covers %d MODIS tile.'%len(tiles.keys()))
        self.mcd43_files = []
        boas, boa_qas, brdf_stds, Hxs, Hys    = [], [], [], [], []
        for key in tiles.keys()[1:]:
            self.logger.info('Getting BOA from MODIS tile: %s.'%key)
            mcd43_file  = glob(self.mcd43_tmp%(self.mcd43_dir, self.year, self.doy, key))[0]
            self.mcd43_files.append(mcd43_file)
            self.H_inds, self.L_inds = tiles[key]
            Lx, Ly = self.L_inds
            Hx, Hy = self.H_inds
            Hxs.append(Hx); Hys.append(Hy)

            vza, sza = self.vza[:, Hx, Hy], self.sza[:, Hx, Hy]
            vaa, saa = self.vaa[:, Hx, Hy], self.saa[:, Hx, Hy]
            raa      = vaa - saa
            boa, boa_qa, brdf_std = get_brdf_six(mcd43_file, angles=[vza, sza, raa],\
                                                 bands=(3,4,1,2,6,7), Linds= [Lx, Ly])
            boas.append(boa); boa_qas.append(boa_qa); brdf_stds.append(brdf_std)

        self.boa    = np.hstack(boas)
        self.boa_qa = np.hstack(boa_qas)
        self.brdf_stds = np.hstack(brdf_stds)
        self.logger.info('Applying spectral transform.')
        self.boa = self.boa*np.array(self.spectral_transform)[0][...,None] + \
                            np.array(self.spectral_transform)[1][...,None]
        self.Hx  = np.hstack(Hxs)
        self.Hy  = np.hstack(Hys)
        
    def solving_l8_aerosol(self,):
        self.logger = logging.getLogger('Landsat 8 Atmospheric Correction')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.logger.propagate = False

        self.sensor  = 'OLI'
        self.logger.info('Doing Landsat 8 tile: (%s, %s) on %d-%02d-%02d.' \
                          % (self.l8_tile[0], self.l8_tile[1], self.year, self.month, self.day))
        self._l8_aerosol()

if __name__ == '__main__':
    aero = solve_aerosol(2017, 4, 21, l8_tile = (123, 34), mcd43_dir   = '/home/ucfafyi/DATA/S2_MODIS/m_data/')
    aero.solving_l8_aerosol()
