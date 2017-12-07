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
from grab_brdf import MCD43_SurRef
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
        
    def _save_img(self, fnames, refs, example_file):
        g            = gdal.Open(example_file)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        bands_refs   = zip(fnames, refs)
        f            = lambda band_ref: self._save_band(band_ref, projection = projection, geotransform = geotransform)
        parmap(f, bands_refs)

    def _save_band(self, band_ref, projection, geotransform):
        fname, ref = band_ref
        nx, ny = ref.shape
        dst_ds = gdal.GetDriverByName('GTiff').Create(fname, ny, nx, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).WriteArray(ref)
        dst_ds.FlushCache()
        dst_ds = None

    def _l8_aerosol(self,):
        self.logger.propagate = False
        self.logger.info('Start to retrieve atmospheric parameters.')
        l8 = read_l8(self.l8_toa_dir, self.l8_tile, self.year, self.month, self.day, bands = self.bands)
        l8._get_angles()
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        self.logger.info('Find corresponding pixels between L8 and MODIS tiles')
        self.example_file = self.l8_toa_dir + '/%s_b%d.tif'%(l8.header, 1)
        if len(glob(self.l8_toa_dir + '/MCD43_%s.npz'%(l8.header))) == 0:
            boa, unc, hx, hy, lx, ly, flist = MCD43_SurRef(self.mcd43_dir, self.example_file, \
                                                           self.year, self.doy, [l8.saa_sza, l8.vaa_vza], 
                                                           sun_view_ang_scale=[0.01, 0.01], bands = [3,4,1,2,6,7], tolz=0.003)
            np.savez(self.l8_toa_dir + 'MCD43_%s.npz'%l8.header, boa=boa, unc=unc, hx=hx, hy=hy, lx=lx, ly=ly, flist=flist) 
        else:
            f = np.load(self.l8_toa_dir + 'MCD43_%s.npz'%l8.header)
            boa, unc, hx, hy, lx, ly, flist = f['boa'], f['unc'], f['hx'], f['hy'], f['lx'], f['ly'], f['flist']
        self.logger.info('Applying spectral transform.')
        self.boa = boa*np.array(self.spectral_transform)[0][...,None, None] + \
                       np.array(self.spectral_transform)[1][...,None, None]
        self.logger.info('Reading in TOA reflectance.')
        toa           = l8._get_toa()
        self.sen_time = l8.sen_time

        self.logger.info('Getting elevation.')
        ele_data = reproject_data(self.global_dem, self.example_file, outputType = gdal.GDT_Float32).data/1000.
        mask = ~np.isfinite(ele_data)
        self.elevation = np.ma.array(ele_data, mask = mask)
        
        self.logger.info('Getting pripors from ECMWF forcasts.')
        aot, tcwv, tco3 = np.array(self._read_cams(self.example_file))
        self.aot        = aot #[self.Hx, self.Hy] #* (1-0.14) # validation of +14% biase
        self.tco3       = tco3#[self.Hx, self.Hy] #* (1 - 0.05)
        self.tcwv       = tcwv#[self.Hx, self.Hy]
        self.aot_unc    = np.ones(self.aot.shape)  * 0.5
        self.tcwv_unc   = np.ones(self.tcwv.shape) * 0.2
        self.tco3_unc   = np.ones(self.tco3.shape) * 0.2

        self.logger.info('Trying to get the aod from ddv method.')
        self._get_ddv_aot(toa, l8, tcwv, tco3, ele_data)
        self.logger.info('Applying PSF model.')
        if self.l8_psf is None:
            self.logger.info('No PSF parameters specified, start solving.')
            high_indexs   = np.where((~toa[-2].mask[::10,::10]) & (~np.isnan(self.boa[-2])[::10,::10]))
            self.high_img = toa[-2][::10,::10]
            self.high_indexs = high_indexs
            low_img     = np.ma.array(self.boa[-2][::10,::10][high_indexs[0], high_indexs[1]])
            qa, cloud   = self.unc[-2][::10,::10][high_indexs[0], high_indexs[1]], l8.qa_mask[::10,::10]
            #toa[-1][~l8.qa_mask] = np.nan
            psf         = psf_optimize(toa[-2][::10,::10].data, high_indexs, low_img, qa, cloud, qa_thresh=0.08, xstd=12., ystd= 20., \
                                       scale = self.spectral_transform[0][-2], offset=self.spectral_transform[1][-2])
            xs, ys      = psf.fire_shift_optimize()
            xstd, ystd  = 12., 20.
            ang         = 0
            self.logger.info('Solved PSF parameters are: %.02f, %.02f, %d, %d, %d, and the correlation is: %f.' \
                                 %(xstd, ystd, 0, xs, ys, 1-psf.costs.min()))
        else:
            xstd, ystd, ang, xs, ys = self.l8_psf
 



    def _get_ddv_aot(self, toa, l8, tcwv, tco3, ele_data):
	ndvi_mask = (((toa[3] - toa[2])/(toa[3] + toa[2])) > 0.6) & (toa[5] > 0.01) & (toa[5] < 0.25)
	if ndvi_mask.sum() < 100:
	    self.logger.info('No enough DDV found in this sence for aot restieval, and only cams prediction used.') 
	else:
	    Hx, Hy = np.where(ndvi_mask)
            if ndvi_mask.sum() > 1000: 
	        random_choice     = np.random.choice(len(Hx), 1000, replace=False)
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
	    rep_g   = reproject_data(g, example_file, outputType = gdal.GDT_Float32).g
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
    aero = solve_aerosol(2017, 7, 10, l8_tile = (123, 34), mcd43_dir   = '/data/selene/ucfajlg/Hebei/MCD43/')
    aero.solving_l8_aerosol()
