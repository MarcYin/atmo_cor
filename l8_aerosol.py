#/usr/bin/env python 
import os
import sys
sys.path.insert(0, 'python')
import gdal
import json
import datetime
import logging
import numpy as np
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
        self.l8_u_bands  = 'band2', 'band3', 'band4', 'band5', 'band6', 'band7' #bands used for the atmo-cor
        self.band_indexs = [2, 3, 4, 5, 6, 7]
        self.boa_bands   = [469, 555, 645, 869, 1640, 2130]
        self.aero_res    = aero_res
        self.mcd43_tmp   = '%s/MCD43A1.A%d%03d.%s.006.*.hdf'
        self.l8_spectral_transform = [[ 1.06946607,  1.03048916,  1.04039226,  1.00163932,  1.00010918, 0.95607606,  0.99951677],
                                      [ 0.0035921 , -0.00142761, -0.00383504, -0.00558762, -0.00570695, 0.00861192,  0.00188871]]
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


    def _s2_aerosol(self,):
        self.logger.propagate = False
        self.logger.info('Start to retrieve atmospheric parameters.')
        self.
        self.logger.info('Reading in TOA reflectance.')
        selected_img = self.s2.get_s2_toa()
        self.s2.get_s2_cloud()
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        #self.s2.cloud[:] = False # due to the bad cloud algrithm 
        self.logger.info('Find corresponding pixels between S2 and MODIS tiles')
        tiles = Find_corresponding_pixels(self.s2.s2_file_dir+'/B04.jp2', destination_res=500)
        if len(tiles.keys())>1:
            self.logger.info('This sentinel 2 tile covers %d MODIS tile.'%len(tiles.keys()))
        self.mcd43_files = []
        boas, boa_qas, brdf_stds, Hxs, Hys    = [], [], [], [], []
        for key in tiles.keys():
            #h,v = int(key[1:3]), int(key[-2:])
            self.logger.info('Getting BOA from MODIS tile: %s.'%key)
            mcd43_file  = glob(self.mcd43_tmp%(self.mcd43_dir, self.year, self.doy, key))[0]
            self.mcd43_files.append(mcd43_file)
            self.H_inds, self.L_inds = tiles[key]
            Lx, Ly = self.L_inds
            Hx, Hy = self.H_inds
            Hxs.append(Hx); Hys.append(Hy)
            self.logger.info( 'Getting the angles and simulated surface reflectance.')
            self.s2.get_s2_angles(self.reconstruct_s2_angle)

            if self.reconstruct_s2_angle:
                self.s2_angles = np.zeros((4, 6, len(Hx)))
                hx, hy = (Hx*23./self.full_res[0]).astype(int), \
                         (Hy*23./self.full_res[1]).astype(int) # index the 23*23 sun angles
                for j, band in enumerate (self.s2_u_bands[:-2]):
                    vhx, vhy = (1.*Hx*self.s2.angles['vza'][band].shape[0]/self.full_res[0]).astype(int), \
                               (1.*Hy*self.s2.angles['vza'][band].shape[1]/self.full_res[1]).astype(int)
                    self.s2_angles[[0,2],j,:] = (self.s2.angles['vza'][band].astype(float)/100.)[vhx, vhy], \
                                                (self.s2.angles['vaa'][band].astype(float)/100.)[vhx, vhy]

                    self.s2_angles[[1,3],j,:] = self.s2.angles['sza'][hx, hy], \
                                                self.s2.angles['saa'][hx, hy]
            else:
                self.s2_angles = np.zeros((4, 6, len(Hx)))
                hx, hy = (Hx*23./self.full_res[0]).astype(int), \
                         (Hy*23./self.full_res[0]).astype(int) # index the 23*23 sun angles
                for j, band in enumerate (self.s2_u_bands[:-2]):
                    self.s2_angles[[0,2],j,:] = self.s2.angles['vza'][band][hx, hy], \
                                                self.s2.angles['vaa'][band][hx, hy]
                    self.s2_angles[[1,3],j,:] = self.s2.angles['sza'][hx, hy], \
                                                self.s2.angles['saa'][hx, hy]

            #use mean value to fill bad values
            for i in range(4):
                mask = ~np.isfinite(self.s2_angles[i])
                if mask.sum()>0:
                    self.s2_angles[i][mask] = np.interp(np.flatnonzero(mask), \
                                                        np.flatnonzero(~mask), \
                                                        self.s2_angles[i][~mask]) # simple interpolation
            vza, sza = self.s2_angles[:2]
            vaa, saa = self.s2_angles[2:]
            raa      = vaa - saa
            # get the simulated surface reflectance
            s2_boa, s2_boa_qa, brdf_std = get_brdf_six(mcd43_file, angles=[vza, sza, raa],\
                                                       bands=(3,4,1,2,6,7), Linds= [Lx, Ly])
            boas.append(s2_boa); boa_qas.append(s2_boa_qa); brdf_stds.append(brdf_std)

        self.s2_boa    = np.hstack(boas)
        self.s2_boa_qa = np.hstack(boa_qas)
        self.brdf_stds = np.hstack(brdf_stds)
        self.logger.info('Applying spectral transform.')
        self.s2_boa = self.s2_boa*np.array(self.s2_spectral_transform)[0,:-1][...,None] + \
                                  np.array(self.s2_spectral_transform)[1,:-1][...,None]
        self.Hx  = np.hstack(Hxs)

