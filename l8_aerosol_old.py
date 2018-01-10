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

from scipy.stats import linregress

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

    def _extend_vals(self, val):
        self.block_size = int(self.aero_res / 30.)
        self.num_blocks = int(np.ceil(max(self.full_res) / self.block_size))
        self.efull_res  = self.block_size * self.num_blocks 
        temp            = np.zeros((self.efull_res, self.efull_res))
        temp[:]         = np.nan
        temp[:self.full_res[0], :self.full_res[1]] = val
        return temp

    def _l8_aerosol(self,):
        self.logger.propagate = False
        self.logger.info('Start to retrieve atmospheric parameters.')
        l8 = read_l8(self.l8_toa_dir, self.l8_tile, self.year, self.month, self.day, bands = self.bands)
        l8._get_angles()
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

            vza, sza = l8.vza[:, Hx, Hy], l8.sza[:, Hx, Hy]
            vaa, saa = l8.vaa[:, Hx, Hy], l8.saa[:, Hx, Hy]
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
        self.sza      = l8.sza[:, self.Hx, self.Hy]
        self.vza      = l8.vza[:, self.Hx, self.Hy]
        self.saa      = l8.saa[:, self.Hx, self.Hy]
        self.vaa      = l8.vaa[:, self.Hx, self.Hy]
        self.logger.info('Reading in TOA reflectance.')
        toa           = l8._get_toa()
        self.toa      = toa[:, self.Hx, self.Hy]
        self.sen_time = l8.sen_time

        self.logger.info('Getting elevation.')
        ele_data = reproject_data(self.global_dem, self.example_file).data
        mask = ~np.isfinite(ele_data)
        ele_data = np.ma.array(ele_data, mask = mask)/1000.
        
        self.logger.info('Getting pripors from ECMWF forcasts.')
        aot, tcwv, tco3 = np.array(self._read_cams(self.example_file))
        self._get_ddv_aot(toa, l8, tcwv, tco3, ele_data) 
   

        aot, tcwv, tco3 = np.array(self._read_cams(self.example_file))
        self.aot        = aot [self.Hx, self.Hy] #* (1-0.14) # validation of +14% biase
        self.tco3       = tco3[self.Hx, self.Hy] #* (1 - 0.05)
        self.tcwv       = tcwv[self.Hx, self.Hy]
        self.aot_unc    = np.ones(self.aot.shape)  * 0.5
        self.tcwv_unc   = np.ones(self.tcwv.shape) * 0.2
        self.tco3_unc   = np.ones(self.tco3.shape) * 0.2



    def _get_ddv_aot(self, toa, l8, tcwv, tco3, ele_data):
	ndvi_mask = (((toa[5] - toa[2])/(toa[5] + toa[2])) > 0.5) & (toa[5] > 0.01) & (toa[5] < 0.25)
	if ndvi_mask.sum() < 100:
	    self.logger.info('No enough DDV found in this sence for aot restieval, and only cams prediction used.') 
	else:
	    Hx, Hy = np.where(ndvi_mask)
            if ndvi_mask.sum() > 25000000: 
	        random_choice     = np.random.choice(len(Hx), 25000000, replace=False)
	        random_choice.sort()
	        Hx, Hy            = Hx[random_choice], Hy[random_choice]
	        ndvi_mask[:]      = False
	        ndvi_mask[Hx, Hy] = True
	    Hx, Hy    = np.where(ndvi_mask)
	    blue_vza  = np.cos(np.deg2rad(l8.vza[0, Hx, Hy]))
            blue_sza  = np.cos(np.deg2rad(l8.sza[0, Hx, Hy]))
	    red_vza   = np.cos(np.deg2rad(l8.vza[2, Hx, Hy])) 
            red_sza   = np.cos(np.deg2rad(l8.sza[2, Hx, Hy]))
	    blue_raa  = np.cos(np.deg2rad(l8.vaa[0, Hx, Hy] - l8.saa[0, Hx, Hy]))
	    red_raa   = np.cos(np.deg2rad(l8.vaa[2, Hx, Hy] - l8.saa[2, Hx, Hy]))
	    red, blue = toa[2, Hx, Hy], toa[0, Hx, Hy]
	    swif      = toa[5, Hx, Hy]
	    red_emus  = np.array(self.emus)[:, 3]
	    blue_emus = np.array(self.emus)[:, 1]

	    zero_aod    = np.zeros_like(red)
	    red_inputs  = np.array([red_sza,  red_vza,  red_raa,  zero_aod, tcwv[Hx, Hy], tco3[Hx, Hy], ele_data[Hx, Hy]])
	    blue_inputs = np.array([blue_sza, blue_vza, blue_raa, zero_aod, tcwv[Hx, Hy], tco3[Hx, Hy], ele_data[Hx, Hy]])
	    
	    p           = np.r_[np.arange(0, 1., 0.02), np.arange(1., 1.5, 0.05),  np.arange(1.5, 2., 0.1)]
	    f           =  lambda aot: self._ddv_cost(aot, blue, red, swif, blue_inputs, red_inputs,  blue_emus, red_emus)
	    costs       = parmap(f, p)
	    min_ind     = np.argmin(costs)
	    self.logger.info('DDV solved aod is %.02f, and it will used as the mean value of cams prediction.'% p[min_ind])
            self.aot   += (p[min_ind] - self.aot.mean())
            #self.costs = costs
            #self.p     = p

    def _ddv_cost(self, aot, blue, red, swif, blue_inputs, red_inputs,  blue_emus, red_emus):
        blue_inputs[3, :] = aot
        red_inputs [3, :] = aot
        blue_xap_emu, blue_xbp_emu, blue_xcp_emu = blue_emus
        red_xap_emu,  red_xbp_emu,  red_xcp_emu  = red_emus
        blue_xap, blue_xbp, blue_xcp             = blue_xap_emu.predict(blue_inputs.T)[0], \
                                                   blue_xbp_emu.predict(blue_inputs.T)[0], \
                                                   blue_xcp_emu.predict(blue_inputs.T)[0]
        red_xap,  red_xbp,  red_xcp              = red_xap_emu.predict(red_inputs.T)  [0], \
                                                   red_xbp_emu.predict(red_inputs.T)  [0], \
                                                   red_xcp_emu.predict(red_inputs.T)  [0]
        y        = blue_xap * blue - blue_xbp
        blue_sur = y / (1 + blue_xcp * y)
        y        = red_xap * red - red_xbp
        red_sur  = y / (1 + red_xcp * y)
        blue_dif = (blue_sur - 0.25 * swif)**2
        red_dif  = (red_sur  - 0.5  * swif)**2
        cost     = 0.5 * (blue_dif + red_dif)
        return cost.sum()

    def _read_cams(self, example_file, parameters = ['aod550', 'tcwv', 'gtco3'], this_scale=[1., 0.1, 46.698]):
	netcdf_file = datetime.datetime(self.sen_time.year, self.sen_time.month, \
					self.sen_time.day).strftime("%Y-%m-%d.nc")
	template    = 'NETCDF:"%s":%s'
	ind         = np.abs((self.sen_time.hour  + self.sen_time.minute/60. + \
			      self.sen_time.second/3600.) - np.arange(0,25,3)).argmin()
	sr         = osr.SpatialReference()
	sr.ImportFromEPSG(4326)
	proj       = sr.ExportToWkt()
	results = []
	for i, para in enumerate(parameters):
	    fname   = template%(self.cams_dir + '/' + netcdf_file, para)
	    g       = gdal.Open(fname)
	    g.SetProjection(proj)
	    sub     = g.GetRasterBand(ind+1)
	    offset  = sub.GetOffset()
	    scale   = sub.GetScale()
	    bad_pix = int(sub.GetNoDataValue())
	    rep_g   = reproject_data(g, example_file).g
	    data    = rep_g.GetRasterBand(ind+1).ReadAsArray()
	    data    = data*scale + offset
	    mask    = (data == (bad_pix*scale + offset))
	    if mask.sum()>=1:
		data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
	    results.append(data*this_scale[i])
        return results

        
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
    aero = solve_aerosol(2017, 7, 10, l8_tile = (123, 34), mcd43_dir   = '/home/ucfafyi/DATA/S2_MODIS/m_data/')
    aero.solving_l8_aerosol()
