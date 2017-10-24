#/usr/bin/env python 
import os
import sys
sys.path.insert(0, 'python')
import gdal
import json
import datetime
import logging
import numpy as np
from glob import glob
from scipy import signal, ndimage
import cPickle as pkl
from smoothn import smoothn
from grab_s2_toa import read_s2
from multi_process import parmap
from reproject import reproject_data
from get_brdf import get_brdf_six
from atmo_paras_optimization import solving_atmo_paras
from spatial_mapping import Find_corresponding_pixels, cloud_dilation
from emulation_engine import AtmosphericEmulationEngine
from psf_optimize import psf_optimize

class solve_aerosol(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,h,v,
                 year, doy,
                 emus_dir    = '/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators',
                 mcd43_dir   = '/data/selene/ucfajlg/Ujia/MCD43/',
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 l8_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/l_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 wv_emus_dir = '/home/ucfafyi/DATA/Multiply/Atmo_cor/data/wv_msi_retrieval_NEW.pkl',
                 cams_dir    = '/home/ucfafyi/DATA/Multiply/cams/',
                 s2_tile     = '29SQB',
                 l8_tile     = (204,33),
                 s2_psf      = [29.75, 39 , 0, 38, 40],
                 l8_psf      = None,
                 qa_thresh   = 255,
                 verbose     = True,
                 aero_res    = 3050, # resolution for aerosol retrival in meters should be larger than 500
                 reconstruct_s2_angle = True):

        self.year        = year 
        self.doy         = doy
        self.date        = datetime.datetime(self.year, 1, 1) \
                                             + datetime.timedelta(self.doy - 1)
        self.month       = self.date.month
        self.day         = self.date.day
        self.h           = h
        self.v           = v
        self.mcd43_dir   = mcd43_dir
        self.emus_dir    = emus_dir
        self.qa_thresh   = qa_thresh
        self.s2_toa_dir  = s2_toa_dir
        self.l8_toa_dir  = l8_toa_dir
        self.global_dem  = global_dem
        self.wv_emus_dir = wv_emus_dir
        self.cams_dir    = cams_dir
        self.s2_tile     = s2_tile
        self.l8_tile     = l8_tile
        self.s2_psf      = s2_psf 
        self.s2_u_bands  = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A', 'B09' #bands used for the atmo-cor
        self.s2_full_res = (10980, 10980)
        self.s_subsample = 1
        self.aero_res    = 3050
        mcd43_tmp        = '%s/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'
        self.mcd43_file  = glob(mcd43_tmp%(self.mcd43_dir,\
                                 self.year, self.doy, self.h, self.v))[0]

        self.reconstruct_s2_angle  = reconstruct_s2_angle
        self.s2_spectral_transform = [[ 1.06946607,  1.03048916,  1.04039226,  1.00163932,  1.00010918, 0.95607606,  0.99951677],
                                      [ 0.0035921 , -0.00142761, -0.00383504, -0.00558762, -0.00570695, 0.00861192,  0.00188871]]       
       
    def _load_emus(self, sensor):
        AEE = AtmosphericEmulationEngine(sensor, self.emus_dir)
        up_bounds   = AEE.emulators[0].inputs[:,4:7].max(axis=0)
        low_bounds  = AEE.emulators[0].inputs[:,4:7].min(axis=0)
        bounds = np.array([low_bounds, up_bounds]).T
        return AEE, bounds

    def repeat_extend(self,data, shape=(10980, 10980)):
        da_shape    = data.shape
        re_x, re_y  = int(1.*shape[0]/da_shape[0]), int(1.*shape[1]/da_shape[1])
        new_data    = np.zeros(shape)
        new_data[:] = -9999
        new_data[:re_x*da_shape[0], :re_y*da_shape[1]] = np.repeat(np.repeat(data, re_x,axis=0), re_y, axis=1)
        return new_data
        
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
        
        self.s2_logger.propagate = False
        self.s2_logger.info('Start to retrieve atmospheric parameters')
        self.s2 = read_s2(self.s2_toa_dir, self.s2_tile, self.year, self.month, self.day, self.s2_u_bands)
	selected_img = self.s2.get_s2_toa() 
	self.s2.get_s2_cloud()
        self.s2.cloud[:] = False # due to the bad cloud algrithm 
        self.s2_logger.info('Find corresponding pixels between S2 and MODIS tiles')
        tiles = Find_corresponding_pixels(self.s2.s2_file_dir+'/B04.jp2', destination_res=500) 
        self.H_inds, self.L_inds = tiles['h%02dv%02d'%(self.h, self.v)]
	self.Lx, self.Ly = self.L_inds
	self.Hx, self.Hy = self.H_inds

        self.s2_logger.info( 'Getting the angles and simulated surface reflectance')
        self.s2.get_s2_angles(self.reconstruct_s2_angle, slic = [self.Hx, self.Hy])
        if self.reconstruct_s2_angle:
	    self.s2_angles = np.zeros((4, 6, len(self.Hx)))
            hx, hy = (self.Hx*23/10980.).astype(int), (self.Hy*23/10980.).astype(int) # index the 23*23 sun angles
            for j, band in enumerate (self.s2_u_bands[:-2]):
                self.s2_angles[[0,2],j,:] = self.s2.angles['vza'][band]/100.,  self.s2.angles['vaa'][band]/100. 
                self.s2_angles[[1,3],j,:] = self.s2.angles['sza'][hx, hy],self.s2.angles['saa'][hx, hy]

        else:
            self.s2_angles = np.zeos((4, 6, len(self.Hx)))
            self.s2_angles[[1,-1],...] = self.s2.angles['msz'], self.s2.angles['msa']
            for i, angle in [[0,'mvz'], [2,'mva']]:
                for j, band in enumerate (self.s2_u_bands[:-2]):
                    self.s2_angles[i][j] = self.s2.angles[angle][band]

        #use mean value to fill bad values
        for i in range(4):
            m = ~np.isfinite(self.s2_angles[i])
            self.s2_angles[i][m] = np.nanmean(self.s2_angles[i])
        vza, sza = self.s2_angles[:2]
	raa      = self.s2_angles[2] - self.s2_angles[3]

        # get the simulated surface reflectance
        self.s2_boa, self.s2_boa_qa,self.brdf_stds = get_brdf_six(self.mcd43_file, angles=[vza, sza, raa],\
                                                                  bands=(3,4,1,2,6,7), Linds= [self.Lx, self.Ly])
        #self.s2_boa, self.s2_boa_qa = self.s2_boa.flatten(), self.s2_boa_qa.flatten()
        self.s2_logger.info('Applying spectral transform.')
        self.s2_boa = self.s2_boa*np.array(self.s2_spectral_transform)[0,:-1][...,None] + \
                                  np.array(self.s2_spectral_transform)[1,:-1][...,None]
        self.s2_logger.info('Getting elevation.')
        ele = reproject_data(self.global_dem, self.s2.s2_file_dir+'/B04.jp2')
        ele.get_it()
        mask = ~np.isfinite(ele.data)
        ele.data = np.ma.array(ele.data, mask = mask)
        self.elevation = ele.data[self.Hx, self.Hy]/1000.
        
        self.s2_logger.info('Getting pripors from ECMWF forcasts.')
	sen_time_str = json.load(open(self.s2.s2_file_dir+'/tileInfo.json', 'r'))['timestamp']
	sen_time     = datetime.datetime.strptime(sen_time_str, u'%Y-%m-%dT%H:%M:%S.%fZ') 
	ind          = np.abs((sen_time.hour+sen_time.minute/60.+sen_time.second/3600.)-np.arange(0,25,3)).argmin() 
	hour         = np.arange(0,25,3)[ind]
	ecmwf_time   = datetime.datetime(sen_time.year, sen_time.month,sen_time.day, hour).strftime("%Y%m%dT%H%M%S")
	aod_file     = '/aod550_' + ecmwf_time + '_global.tif'
	tco3_file    = '/tco3_'   + ecmwf_time + '_global.tif'

	aod550         = reproject_data(self.cams_dir + aod_file, self.s2.s2_file_dir+'/B04.jp2')
        aod550.get_it()
        aod_scale      = float(aod550.g.GetMetadata()['aod550#scale_factor'])
        aod_offset     = float(aod550.g.GetMetadata()['aod550#add_offset'])
        self.s2_aod550 = (aod550.data*aod_scale + aod_offset)[self.Hx, self.Hy] * (1-0.14) # validation of +14% biase
        mask           = (self.s2_aod550 <= 0.00045)
        if mask.sum() >= 1:
            self.s2_aod550[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.s2_aod550[~mask])
	tco3           = reproject_data(self.cams_dir + tco3_file, self.s2.s2_file_dir+'/B04.jp2')
        tco3.get_it()
        tco3_scale     = float(tco3.g.GetMetadata()['gtco3#scale_factor'])
        tco3_offset    = float(tco3.g.GetMetadata()['gtco3#add_offset'])
        self.s2_tco3   = (tco3.data*tco3_scale + tco3_offset)\
                          [self.Hx, self.Hy] * 46.698 * (1 - 0.05) # units and validation of +5% biase
        mask           = (self.s2_tco3 <= 0.152)   
        if mask.sum() >= 1:
            self.s2_tco3[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.s2_tco3[~mask])
        self.s2_tco3_unc   = np.ones(self.s2_tco3.shape)   * 0.2
        self.s2_aod550_unc = np.ones(self.s2_aod550.shape) * 0.5

        self.s2_logger.info('Trying to get the tcwv from the emulation of sen2cor look up table.')
        try:
            b8a, b9  = np.repeat(np.repeat(selected_img['B8A']*0.0001, 2, axis=0), 2, axis=1)[self.Hx, self.Hy],\
                                np.repeat(np.repeat(selected_img['B09']*0.0001, 6, axis=0), 6, axis=1)[self.Hx, self.Hy]
	    wv_emus   = pkl.load(open(self.wv_emus_dir, 'rb'))
	    inputs    = np.array([b9, b8a, vza[-1], sza[-1], abs(raa)[-1], self.elevation]).T
            tcwv_mask = b8a < 0.05 
	    self.s2_tcwv, self.s2_tcwv_unc, _ = wv_emus.predict(inputs, do_unc = True)
            if tcwv_mask.sum() >= 1:
                self.s2_tcwv[tcwv_mask] = np.interp(np.flatnonzero( tcwv_mask), \
                                                    np.flatnonzero(~tcwv_mask), self.s2_tcwv[~tcwv_mask]) # simple interpolation
	except:
            self.s2_logger.warning('Getting tcwv from the emulation of sen2cor look up table failed, ECMWF data used.')
            tcwv_file      = '/tcwv_' + ecmwf_time + '_global.tif'
            tcwv           = reproject_data(self.cams_dir + tcwv_file, self.s2.s2_file_dir+'/B04.jp2')
            tcwv.get_it()
            tcwv_scale     = float(tcwv.g.GetMetadata()['tcwv#scale_factor'])
            tcwv_offset    = float(tcwv.g.GetMetadata()['tcwv#add_offset'])
            self.s2_tcwv   = (tcwv.data*tcwv_scale + tcwv_offset)[self.Hx, self.Hy]/10.
            mask           = (self.s2_tcwv <= 0.0054)
            if mask.sum() >= 1:
               self.s2_tcwv[mask] = np.interp(np.flatnonzero( mask), \
                                              np.flatnonzero(~mask), self.s2_tcwv[~mask])
            self.s2_tcwv_unc = np.ones(self.s2_tcwv.shape) * 0.2

        self.s2_logger.info('Applying PSF model.')
        if self.s2_psf is None:
            self.s2_logger.info('No PSF parameters specified, start solving.')
            high_img    = np.repeat(np.repeat(selected_img['B12'], 2, axis=0), 2, axis=1)*0.0001
            high_indexs = self.H_inds
            low_img     = self.s2_boa[-2]
            qa, cloud   = self.s2_boa_qa[-2], self.s2.cloud
            psf         = psf_optimize(high_img, high_indexs, low_img, qa, cloud, 2)
            xs, ys      = psf.fire_shift_optimize()
            xstd, ystd  = 29.75, 39
            ang         = 0
            self.s2_logger.info('Solved PSF parameters are: %.02f, %.02f, %d, %d, %d, and the correlation is: %f.' \
                                 %(xstd, ystd, 0, xs, ys, 1-psf.costs.min()))
        else:
            xstd, ystd, ang, xs, ys = self.s2_psf
        # apply psf shifts without going out of the image extend  
        shifted_mask = np.logical_and.reduce(((self.Hx+int(xs)>=0),
                                              (self.Hx+int(xs)<self.s2_full_res[0]), 
                                              (self.Hy+int(ys)>=0),
                                              (self.Hy+int(ys)<self.s2_full_res[0])))
        
        self.Hx, self.Hy = self.Hx[shifted_mask]+int(xs), self.Hy[shifted_mask]+int(ys)
        self.Lx, self.Ly = self.Lx[shifted_mask], self.Ly[shifted_mask]
        self.s2_boa      = self.s2_boa[:,shifted_mask]
        self.s2_boa_qa   = self.s2_boa_qa[:, shifted_mask]
        self.s2_angles   = self.s2_angles[:,:,shifted_mask]
        self.elevation   = self.elevation[shifted_mask]
        self.s2_aod550   = self.s2_aod550[shifted_mask]
        self.s2_tcwv     = self.s2_tcwv[shifted_mask]
        self.s2_tco3     = self.s2_tco3[shifted_mask]

        self.s2_aod550_unc = self.s2_aod550_unc[shifted_mask]
        self.s2_tcwv_unc   = self.s2_tcwv_unc[shifted_mask]
        self.s2_tco3_unc   = self.s2_tco3_unc[shifted_mask]
        self.brdf_stds     = self.brdf_stds[:,shifted_mask]

        self.s2_logger.info('Getting the convolved TOA reflectance.')
        self.valid_pixs = sum(shifted_mask) # count how many pixels is still within the s2 tile 
        ker_size        = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        self.bad_pixs   = np.zeros(self.valid_pixs).astype(bool)
        imgs = []
        for i, band in enumerate(self.s2_u_bands[:-2]):
            if selected_img[band].shape != self.s2_full_res:
                selected_img[band] = self.repeat_extend(selected_img[band], shape = self.s2_full_res)
            else:
                pass
            selected_img[band][0,:]  = -9999
            selected_img[band][-1,:] = -9999
            selected_img[band][:,0]  = -9999
            selected_img[band][:,-1] = -9999
            imgs.append(selected_img[band])
            # filter out the bad pixels
            self.bad_pixs |= cloud_dilation(self.s2.cloud |\
                                            (selected_img[band] <= 0) | \
                                            (selected_img[band] >= 10000),\
                                            iteration= ker_size/2)[self.Hx, self.Hy]
        del selected_img; del self.s2.selected_img;
        ker = self.gaussian(xstd, ystd, ang) 
        f   = lambda img: signal.fftconvolve(img, ker, mode='same')[self.Hx, self.Hy]*0.0001 
        self.s2_toa = np.array(parmap(f,imgs))
        del imgs

        # get the valid value masks
        qua_mask = np.all(self.s2_boa_qa <= self.qa_thresh, axis = 0)

        boa_mask = np.all(~self.s2_boa.mask,axis = 0 ) &\
                          np.all(self.s2_boa > 0, axis = 0) &\
                          np.all(self.s2_boa < 1, axis = 0)
        toa_mask = (~self.bad_pixs) &\
                    np.all(self.s2_toa > 0, axis = 0) &\
                    np.all(self.s2_toa < 1, axis = 0)
        self.s2_mask = boa_mask & toa_mask & qua_mask & (~self.elevation.mask)
        self.s2_AEE, self.s2_bounds = self._load_emus(self.s2_sensor)

    def solving_s2_aerosol(self,):
        
        self.s2_logger = logging.getLogger('Sentinel 2 Atmospheric Correction')
        self.s2_logger.setLevel(logging.INFO)
        if not self.s2_logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.s2_logger.addHandler(ch)
        self.s2_logger.propagate = False

        self.s2_sensor = 'MSI'
        self._s2_aerosol()
        self.s2_solved = []
        if self.aero_res < 500:
            self.s2_logger.warning( 'The best resolution of aerosol should be larger \
                                     than 500 meters (inlcude), so it is set to 500 meters.')
            self.aero_res = 500
        self.block_size = int(self.aero_res/10)
        num_blocks = int(np.ceil(10980/self.block_size)) 
        self.s2_logger.info('Start solving...')
        #for i in range(num_blocks):
        #    for j in range(num_blocks):
        #        self.s2_logger.info('Doing block %03d-%03d.'%(i+1,j+1))
        #        self._s2_block_solver([i,j])
        blocks = zip(np.repeat(range(num_blocks), num_blocks), np.tile(range(num_blocks), num_blocks)) 
        self.s2_solved = parmap(aero._s2_block_solver, blocks)   
        inds = np.array([[i[0], i[1]] for i in self.s2_solved])
        rets = np.array([i[2][0]      for i in self.s2_solved])
          
        aod_map    = np.zeros((num_blocks,num_blocks))
        aod_map[:] = np.nan
        tcwv_map   = aod_map.copy()
        tco3_map   = aod_map.copy()

        aod_map [inds[:,0], inds[:,1]] = rets[:,0]
        tcwv_map[inds[:,0], inds[:,1]] = rets[:,1]
        tco3_map[inds[:,0], inds[:,1]] = rets[:,2]
        para_names = 'aod550', 'tcwv', 'tco3'
         
        g = gdal.Open(self.s2.s2_file_dir+'/10meter.vrt')
        xmin, ymax = g.GetGeoTransform()[0], g.GetGeoTransform()[3]
        projection = g.GetProjection()
        
        results = []
        self.s2_logger.info('Finished retrieval and saving them into local files.')
        for i,para_map in enumerate([aod_map, tcwv_map, tco3_map]):
            s     = smoothn(para_map.copy(), isrobust=True, verbose=False)[1]
            smed  = smoothn(para_map.copy(), isrobust=True, verbose=False, s=s)[0]
            xres, yres = self.block_size*10, self.block_size*10
            geotransform = (xmin, xres, 0, ymax, 0, -yres)
            nx, ny = smed.shape
            dst_ds = gdal.GetDriverByName('GTiff').Create(self.s2.s2_file_dir + \
                                          '/%s.tif'%para_names[i], ny, nx, 1, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform)   
            dst_ds.SetProjection(projection) 
            dst_ds.GetRasterBand(1).WriteArray(smed)
            dst_ds.FlushCache()                     
            dst_ds = None
            results.append(smed)
        self.aod550_map, self.tcwv_map, self.tco3_map = results

        # solve by block
    def _s2_block_solver(self, block):
        i,j = block
        self.s2_logger.info('Doing block %03d-%03d.'%(i+1,j+1))
        block_mask = np.logical_and.reduce(((self.Hx >= i*self.block_size),
                                            (self.Hx < (i+1)*self.block_size),
                                            (self.Hy >= j*self.block_size),
                                            (self.Hy < (j+1)*self.block_size)))
        mask      = self.s2_mask[block_mask]
	prior     = self.s2_aod550[block_mask].mean(), \
		    self.s2_tcwv[block_mask].mean(), self.s2_tco3[block_mask].mean()
        if mask.sum() <= 0:
            self.s2_logger.warning('No valid values in block %03d-%03d, and priors are used for this block.'%(i+1,j+1))
            return [i,j,[prior, 0], prior]
        else:
	    boa, toa  = self.s2_boa[:, block_mask], self.s2_toa[:, block_mask]
	    vza, sza  = self.s2_angles[:2,:, block_mask]*np.pi/180. 
	    vaa, saa  = self.s2_angles[2:,:, block_mask]
	    boa_qa    = self.s2_boa_qa[:, block_mask]
	    elevation = self.elevation[block_mask]
            brdf_std  = self.brdf_stds[:,block_mask]   
	    self.atmo = solving_atmo_paras(self.s2_sensor, self.emus_dir, boa, toa, sza, vza, saa, vaa,\
				           elevation, boa_qa, boa_bands=[469, 555, 645, 869, 1640, 2130], mask=mask,\
                                           band_indexs=[1, 2, 3, 7, 11, 12], prior=prior, subsample=1, brdf_std=brdf_std)

	    self.atmo._load_unc()
	    self.atmo.aod_unc   = self.s2_aod550_unc[block_mask].max()
	    self.atmo.wv_unc    = self.s2_tcwv_unc[block_mask].max() * 5 # inflating it..
	    self.atmo.ozone_unc = self.s2_tco3_unc[block_mask].max()
	    self.atmo.AEE       = self.s2_AEE
	    self.atmo.bounds    = self.s2_bounds
            return ([i,j, self.atmo.optimization(), prior])
        

if __name__ == "__main__":
    aero = solve_aerosol(17, 5, 2017, 247,mcd43_dir = '/home/ucfafyi/DATA/Multiply/MCD43/', \
                                   emus_dir = '/home/ucfafyi/DATA/Multiply/', s2_tile='29SQB', s2_psf=None)
    aero.solving_s2_aerosol()
    #solved  = aero.prepare_modis()