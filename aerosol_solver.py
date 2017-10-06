#/usr/bin/env python 
import os
import sys
import gdal
import json
sys.path.insert(0, 'python')
from glob import glob
from get_modis_toa import grab_modis_toa
import cPickle as pkl
from get_brdf import get_brdf_six
import numpy as np
from atmo_cor import atmo_cor
import datetime
from scipy import signal, ndimage
from grab_s2_toa import read_s2
from spatial_mapping import Find_corresponding_pixels, cloud_dilation
from modis_l1b_reader import MODIS_L1b_reader
from emulation_engine import AtmosphericEmulationEngine
from psf_optimize import psf_optimize
from multi_process import parmap
from reproject import reproject_data

class solve_aerosol(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,h,v,
                 year, doy,
                 emus_dir    = '/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators',
                 mcd43_dir   = '/data/selene/ucfajlg/Ujia/MCD43/',
                 mod_l1b_dir = '/data/selene/ucfajlg/Ujia/MODIS_L1b/GRIDDED',
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 l8_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/l_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 wv_emus_dir = '/home/ucfafyi/DATA/Multiply/Atmo_cor/data/wv_msi_retrieval.pkl',
                 cams_dir    = '/home/ucfafyi/DATA/Multiply/cams/',
                 s2_tile     = '29SQB',
                 l8_tile     = (204,33),
                 s2_psf      = [29.75, 39 , 0, 38, 40],
                 l8_psf      = None,
                 qa_thresh   = 255,
                 mod_cloud   = None,
                 verbose     = True,
                 save_file   = False,
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
        self.mod_l1b_dir = mod_l1b_dir
        self.emus_dir    = emus_dir
        self.qa_thresh   = qa_thresh
        self.mod_cloud   = mod_cloud 
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
        self.m_subsample = 10
        self.s_subsample = 1
        self.block_size  = 100
        mcd43_tmp        = '%s/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'
        self.mcd43_file  = glob(mcd43_tmp%(self.mcd43_dir,\
                                 self.year, self.doy, self.h, self.v))[0]

        self.reconstruct_s2_angle = reconstruct_s2_angle
        self.s2_spectral_transform = [[ 1.06946607,  1.03048916,  1.04039226,  1.00163932,  1.00010918, 0.95607606,  0.99951677],
                                      [ 0.0035921 , -0.00142761, -0.00383504, -0.00558762, -0.00570695, 0.00861192,  0.00188871]]       
       
        self.prior        =  0.2, 3.4, 0.35
    def _load_emus(self, sensor):
        AEE = AtmosphericEmulationEngine(sensor, self.emus_dir)
        up_bounds   = AEE.emulators[0].inputs[:,4:7].max(axis=0)
        low_bounds  = AEE.emulators[0].inputs[:,4:7].min(axis=0)
        bounds = np.array([low_bounds, up_bounds]).T
        return AEE, bounds
    def prepare_modis(self,):
        
        modis_l1b       = MODIS_L1b_reader(self.mod_l1b_dir, "h%02dv%02d"%(self.h,self.v),self.year)
        self.modis_files = [modis_l1b.granules[i] for i in modis_l1b.granules.keys() if i.date() == self.date.date()]
        #self.modis_toa, self.modis_angles = grab_modis_toa(year=2006,doy=200,verbose=True,\
        #                                                   mcd43file = self.mcd43_file, directory_l1b= self.mod_l1b_dir)
        for modis_file in self.modis_files[1:2]:
            band_files  = [getattr(modis_file, 'b%d'%band) for band in range(1,8)]
            angle_files = [getattr(modis_file, ang) for ang in ['vza', 'sza', 'vaa', 'saa']]
            modis_toa   = []
            modis_angle = []
            for band_file in band_files:
                g = gdal.Open(band_file)
                if g is None:
                    raise IOError
                else:
                    data = g.ReadAsArray()
                modis_toa.append(data)
            for angle_file in angle_files:
                g = gdal.Open(angle_file)
                if g is None:
                    raise IOError
                else:
                    data = g.ReadAsArray()
                modis_angle.append(data)

            scale  = np.array(modis_file.scale)
            offset = np.array(modis_file.offset)

            self.modis_toa   = np.array(modis_toa)*np.array(scale)[:,None, None] + offset[:,None, None]
            self.modis_angle = np.array(modis_angle)/100.
            #if sensor=='TERRA'
            #     self.modis_sensor = 'TERRA'
            #elif sensor == 'AQUA':
            #     self.modis_sensor = 'AQUA' 
            self.modis_sensor =  'TERRA' # Only TERRA used at the moment
            self.modis_aerosol()
     
    def modis_aerosol(self, save_file=False):
        
        vza, sza, vaa, saa = self.modis_angle
	self.modis_boa, self.modis_boa_qa = get_brdf_six(self.mcd43_file,angles = [vza, sza, vaa - saa],
							 bands= (1,2,3,4,5,6,7), flag=None, Linds= None)
	if self.mod_cloud is None:
	    self.modis_cloud = np.zeros_like(self.modis_toa[0]).astype(bool)

	qua_mask = np.all(self.modis_boa_qa <= self.qa_thresh, axis =0) 
	boa_mask = np.all(~self.modis_boa.mask, axis = 0) &\
	                  np.all(self.modis_boa>0, axis=0) &\
			  np.all(self.modis_boa<1, axis=0)
	toa_mask = np.all(np.isfinite(self.modis_toa), axis=0) &\
                          np.all(self.modis_toa>0, axis=0) & \
                          np.all(self.modis_toa<1, axis=0)
	self.modis_mask = qua_mask & boa_mask & toa_mask & (~self.modis_cloud)
       
        self.modis_AEE, self.modis_bounds = self._load_emus(self.modis_sensor)
        self.modis_solved = []
    def _m_block_solver(self,block):
	i,j = block
	block_mask= np.zeros_like(self.modis_mask).astype(bool)
	block_mask[i*self.block_size:(i+1)*self.block_size,j*self.block_size:(j+1)*self.block_size] = True        
	boa, toa  = self.modis_boa[:, block_mask].reshape(7,self.block_size, self.block_size),\
                    self.modis_toa[:, block_mask].reshape(7,self.block_size, self.block_size)
	vza, sza  = (self.modis_angle[:2, block_mask]*np.pi/180.).reshape(2,self.block_size, self.block_size)
	vaa, saa  = self.modis_angle[2:, block_mask].reshape(2,self.block_size, self.block_size)
	boa_qa    = self.modis_boa_qa[:, block_mask].reshape(7,self.block_size,self.block_size)
	mask      = self.modis_mask[block_mask].reshape(self.block_size, self.block_size)
	prior     = 0.2, 3.4, 0.35
	self.atmo = atmo_cor(self.modis_sensor, self.emus_dir, boa, toa, sza, vza, 
                             saa, vaa,0.5, boa_qa, boa_bands=[645,869,469,555,1240,1640,2130], 
                             band_indexs=[0,1,2,3,4,5,6], mask=mask, prior=prior, subsample=self.m_subsample)
	self.atmo._load_unc()
	self.atmo.AEE    = self.modis_AEE
	self.atmo.bounds = self.modis_bounds
	if mask.sum() <= 0:
	    print 'No valid values in block %03d-%03d'%(i,j)
	else:
            self.modis_solved.append([i,j, self.atmo.optimization()])

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
    
    def s2_aerosol(self,):
        self.s2_sensor = 'MSI'
        self.s2 = read_s2(self.s2_toa_dir, self.s2_tile, self.year, self.month, self.day, self.s2_u_bands)
	selected_img = self.s2.get_s2_toa() 
	self.s2.get_s2_cloud()
        self.s2.cloud[:] = False # due to the bad cloud algrithm 
        # find corresponding pixels between s2 and modis
        tiles = Find_corresponding_pixels(self.s2.s2_file_dir+'/B04.jp2', destination_res=500) 
        self.H_inds, self.L_inds = tiles['h%02dv%02d'%(self.h, self.v)]
	self.Lx, self.Ly = self.L_inds
	self.Hx, self.Hy = self.H_inds

        # prepare for the angles and simulate the surface reflectance
        self.s2.get_s2_angles(self.reconstruct_s2_angle, slic = [self.Hx, self.Hy])
        if self.reconstruct_s2_angle:
	    self.s2_angles = np.zeros((4, 7, len(self.Hx)))
            hx, hy = (self.Hx*23/10980.).astype(int), (self.Hy*23/10980.).astype(int) # index the 23*23 sun angles
            for j, band in enumerate (self.s2_u_bands[:-1]):
                self.s2_angles[[0,2],j,:] = self.s2.angles['vza'][band]/100.,  self.s2.angles['vaa'][band]/100. 
                self.s2_angles[[1,3],j,:] = self.s2.angles['sza'][hx, hy],self.s2.angles['saa'][hx, hy]

        else:
            self.s2_angles = np.zeos((4, 7, len(self.Hx)))
            self.s2_angles[[1,-1],...] = self.s2.angles['msz'], self.s2.angles['msa']
            for i, angle in [[0,'mvz'], [2,'mva']]:
                for j, band in enumerate (self.s2_u_bands[:-1]):
                    self.s2_angles[i][j] = self.s2.angles[angle][band]

        # use mean value to fill bad values
        for i in range(4):
            m = ~np.isfinite(self.s2_angles[i])
            self.s2_angles[i][m] = np.nanmean(self.s2_angles[i])
        vza, sza = self.s2_angles[:2]
	raa      = self.s2_angles[2] - self.s2_angles[3]

        # get the simulated surface reflectance
        self.s2_boa, self.s2_boa_qa = get_brdf_six(self.mcd43_file, angles=[vza, sza, raa],\
                                                   bands=(3,4,1,2,6,7,2), Linds= [self.Lx, self.Ly])
        #self.s2_boa, self.s2_boa_qa = self.s2_boa.flatten(), self.s2_boa_qa.flatten()
        # apply the spectral transform...
        self.s2_boa = self.s2_boa*np.array(self.s2_spectral_transform)[0,None].T + \
                      np.array(self.s2_spectral_transform)[1,None].T

        # get elevation
        ele = reproject_data(self.global_dem, self.s2.s2_file_dir+'/B04.jp2')
        ele.get_it()
        mask = ~np.isfinite(ele.data)
        ele.data = np.ma.array(ele.data, mask = mask)
        self.elevation = ele.data[self.Hx, self.Hy]/1000.
        
        # get pripors from ECMWF forcasts
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

	tco3         = reproject_data(self.cams_dir + tco3_file, self.s2.s2_file_dir+'/B04.jp2')
        tco3.get_it()
        tco3_scale   = float(tco3.g.GetMetadata()['gtco3#scale_factor'])
        tco3_offset  = float(tco3.g.GetMetadata()['gtco3#add_offset'])
        self.s2_tco3 = (tco3.data*tco3_scale + tco3_offset)\
                       [self.Hx, self.Hy] * 46.698 * (1 - 0.05) # units and validation of +5% biase

        self.s2_tco3_unc   = np.ones(self.s2_tco3.shape)   * 0.2
        self.s2_aod550_unc = np.ones(self.s2_aod550.shape) * 0.5

        # try to get the tcwv from the eulation of sen2cor look up table
        try:
            b8a, b9 = np.repeat(np.repeat(selected_img['B8A']*0.0001, 2, axis=0), 2, axis=1)[self.Hx, self.Hy],\
                                np.repeat(np.repeat(selected_img['B09']*0.0001, 6, axis=0), 6, axis=1)[self.Hx, self.Hy]
	    wv_emus = pkl.load(open(self.wv_emus_dir, 'rb'))
	    inputs  = np.array([b9, b8a, vza[-1], sza[-1], abs(raa)[-1], self.elevation]).T
	    self.s2_tcwv, self.s2_tcwv_unc, _ = wv_emus.predict(inputs, do_unc = True) 
	except:
            tcwv_file     = '/tcwv_' + ecmwf_time + '_global.tif'
            tcwv          = reproject_data(self.cams_dir + tcwv_file, self.s2.s2_file_dir+'/B04.jp2')
            tcwv.get_it()
            tcwv_scale    = float(tcwv.g.GetMetadata()['tcwv#scale_factor'])
            tcwv_offset   = float(tcwv.g.GetMetadata()['tcwv#add_offset'])
            self.s2_tcwv  = (tcwv.data*tcwv_scale + tcwv_offset)[self.Hx, self.Hy]
            self.s2_tcwv_unc = np.ones(self.tcwv.shape) * 0.2

        # get the psf parameters
        if self.s2_psf is None:
            high_img    = np.repeat(np.repeat(selected_img['B12'], 2, axis=0), 2, axis=1)*0.0001
            high_indexs = self.H_inds
            low_img     = self.s2_boa[-2]
            qa, cloud   = self.s2_boa_qa[-2], self.s2.cloud
            psf         = psf_optimize(high_img, high_indexs, low_img, qa, cloud, 2)
            xs, ys      = psf.fire_shift_optimize()
            xstd, ystd  = 29.75, 39
            print 'Solved PSF parameters are: ',xstd, ystd, 0, xs, ys, 'and the correlation is: ', 1-self.costs
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

        # get the convolved toa reflectance
        self.valid_pixs = sum(shifted_mask) # count how many pixels is still within the s2 tile 
        ker_size        = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        self.bad_pixs   = np.zeros(self.valid_pixs).astype(bool)
        imgs = []
        for i, band in enumerate(self.s2_u_bands[:-1]):
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

        ker = self.gaussian(xstd, ystd, ang) 
        f   = lambda img: signal.fftconvolve(img, ker, mode='same')[self.Hx, self.Hy]*0.0001        

        self.s2_toa = np.array(parmap(f,imgs))
        del selected_img; #del self.s2.selected_img; del imgs

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
        self.s2_solved = []

        # solve by block
    def _s2_block_solver(self, block):
        i,j = block
        block_mask = np.logical_and.reduce(((self.Hx >= i*self.block_size),
                                            (self.Hx < (i+1)*self.block_size),
                                            (self.Hy >= j*self.block_size),
                                            (self.Hy < (j+1)*self.block_size)))
         
        boa, toa  = self.s2_boa[:, block_mask], self.s2_toa[:, block_mask]
	vza, sza  = self.s2_angles[:2,:, block_mask]*np.pi/180. 

        boa_qa    = self.s2_boa_qa[:, block_mask]
        mask      = self.s2_mask[block_mask]
        elevation = self.elevation[block_mask]
        prior     = self.s2_aod550[block_mask], self.s2_tcwv[block_mask], self.s2_tco3[block_mask]
        self.atmo = atmo_cor(self.s2_sensor, self.emus_dir, boa, toa, sza, vza, saa, vaa,\
                             elevation, boa_qa, boa_bands=[469, 555, 645, 869, 1640, 2130, 869],\
                             band_indexs=[1, 2, 3, 7, 11, 12, 8], mask=mask, prior=prior, subsample=1)

        self.atmo._load_unc()
        self.atmo.aod_unc   = self.s2_aod550_unc[block_mask]
        self.atmo.wv_unc    = self.s2_tcwv_unc[block_mask]
        self.atmo.ozone_unc = self.s2_tco3_unc[block_mask]
        self.atmo.AEE       = self.s2_AEE
        self.atmo.bounds    = self.s2_bounds

        if mask.sum() <= 0:
            print 'No valid values in block %03d-%03d'%(i,j)
        else:
            self.s2_solved.append([i,j, self.atmo.optimization()])

if __name__ == "__main__":
    aero = solve_aerosol(17,5,2017, 230)
    #aero.s2_aerosol()
    solved  = aero.prepare_modis()
