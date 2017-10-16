#/usr/bin/env python
import sys
sys.path.insert(0,'python')
import gdal
import numpy as np
import logging
from Py6S import *
from multi_process import parmap
from grab_s2_toa import read_s2
from aerosol_solver import solve_aerosol
from reproject import reproject_data
from emulation_engine import AtmosphericEmulationEngine

class atmospheric_correction(object):
    '''
    A class doing the atmospheric coprrection with the input of TOA reflectance
    angles, elevation and emulators of 6S from TOA to surface reflectance.
    '''
    def __init__(self,
                 year, 
                 month, 
                 day,
                 s2_tile,
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 inverse_emu = '/home/ucfafyi/DATA/Multiply/inverse_emus/'):              
        
        self.year        = year
        self.month       = month
        self.day         = day
        self.s2_tile     = s2_tile
        self.s2_toa_dir  = s2_toa_dir
        self.global_dem  =  global_dem
        self.inverse_emu = inverse_emu

        # create logger
	self.logger = logging.getLogger('Sentinel 2 Atmospheric Correction')
	self.logger.setLevel(logging.INFO)
	# create console handler and set level to debug
        if not self.logger.handlers:       
	    ch = logging.StreamHandler()
	    ch.setLevel(logging.DEBUG)
	    # create formatter
	    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	    # add formatter to ch
	    ch.setFormatter(formatter)
	    # add ch to logger
	    self.logger.addHandler(ch)

	    # 'application' code
	    #logger.debug('debug message')
	    #logger.info('info message')
	    #logger.warn('warn message')
	    #logger.error('error message')
	    #logger.critical('critical message')

    def _load_inverse_emus(self, sensor):
	AEE = AtmosphericEmulationEngine(sensor, self.inverse_emu)
        return AEE

    def atmospheric_correction(self,):

        self.logger.propagate = False
        self.s2_sensor = 'MSI'
        self.logger.info('Loading emulators.')
        self.s2_inv_AEE = self._load_inverse_emus(self.s2_sensor)
        self.s2   = read_s2(self.s2_toa_dir, self.s2_tile, \
                            self.year, self.month, self.day, bands=None)
        self.logger.info('Reading in the reflectance.')
        all_refs = self.s2.get_s2_toa()
        self.logger.info('Reading in the angles')
        self.s2.get_s2_angles()
        all_angs = self.s2.angles
        self.sza,self.saa = all_angs['sza'], all_angs['saa']
        
        self.logger.info('Doing 10 meter bands')
        self._10meter_ref = np.array([all_refs[band]/10000. for band \
                                      in ['B02', 'B03', 'B04', 'B08']])
        self._10meter_vza = np.array([all_angs['vza'][band]/100. for band
                                      in ['B02', 'B03', 'B04', 'B08']])
        self._10meter_vaa = np.array([all_angs['vaa'][band]/100. for band
                                      in ['B02', 'B03', 'B04', 'B08']])
        self._10meter_sza = np.repeat(np.repeat(self.sza, 10980/23+1, axis=0), 10980/23+1, axis=1)[:10980, :10980]
        self._10meter_saa = np.repeat(np.repeat(self.saa, 10980/23+1, axis=0), 10980/23+1, axis=1)[:10980, :10980]

        self.logger.info('Getting control variables for 10 meters bands.')
        self._10meter_aod, self._10meter_tcwv, self._10meter_tco3,\
                           self._10meter_ele = self.get_control_variables('B04')
        self.block_size = 183
        self.num_blocks = 10980/183
        self._10meter_band_indexs = [1, 2, 3, 7]        
        self.rsr = [PredefinedWavelengths.S2A_MSI_02, PredefinedWavelengths.S2A_MSI_03, \
                    PredefinedWavelengths.S2A_MSI_04, PredefinedWavelengths.S2A_MSI_08 ]
        self.logger.info('Fire correction.')
        self.fire_correction(self._10meter_ref, self._10meter_sza, self._10meter_vza,\
                             self._10meter_saa, self._10meter_vaa, self._10meter_aod,\
                             self._10meter_tcwv, self._10meter_tco3, self._10meter_ele,\
                             self._10meter_band_indexs)       

        self.logger.info('Doing 20 meter bands')
        self._20meter_ref = [all_refs[band]/10000. for band \
                             in ['B05', 'B05', 'B07', 'B8A', 'B11', 'B12']]
        self._20meter_vza = [all_angs['vza'][band]/100. for band
                             in ['B05', 'B05', 'B07', 'B8A', 'B11', 'B12']]
        self._20meter_vaa = [all_angs['vaa'][band]/100. for band
                             in ['B05', 'B05', 'B07', 'B8A', 'B11', 'B12']]
        self._20meter_sza = np.repeat(np.repeat(self.sza, 5490/23+1, axis=0), 5490/23, axis=1)[:5490, :5490]
        self._20meter_saa = np.repeat(np.repeat(self.saa, 5490/23+1, axis=0), 5490/23, axis=1)[:5490, :5490]

        self.logger.info('Getting control variables for 20 meters bands.')
        self.get_control_variables('B05')



        self.logger.info('Doing 60 meter bands')
        self._60meter_ref = [all_refs[band]/10000. for band \
                             in ['B01', 'B09', 'B10']]
        self._60meter_vza = [all_angs['vza'][band]/100. for band
                             in ['B01', 'B09', 'B10']]
        self._60meter_vaa = [all_angs['vaa'][band]/100. for band
                             in ['B01', 'B09', 'B10']]
        self._60meter_sza = np.repeat(np.repeat(self.sza, 1830/23+1, axis=0), 1830/23, axis=1)[:1830, :1830]
        self._60meter_saa = np.repeat(np.repeat(self.saa, 1830/23+1, axis=0), 1830/23, axis=1)[:1830, :1830]

        self.logger.info('Getting control variables for 60 meters bands.')
        self.get_control_variables('B09')



        del all_refs; del self.s2.selected_img; del all_angs; del self.s2.angles
               


    def get_control_variables(self, target_band):

	aod = reproject_data(self.s2.s2_file_dir+'/aod550.tif', \
                             self.s2.s2_file_dir+'/%s.jp2'%target_band)
        aod.get_it()

        tcwv = reproject_data(self.s2.s2_file_dir+'/tcwv.tif', \
                              self.s2.s2_file_dir+'/%s.jp2'%target_band)
        tcwv.get_it()

        tco3 = reproject_data(self.s2.s2_file_dir+'/tco3.tif', \
                              self.s2.s2_file_dir+'/%s.jp2'%target_band)
        tco3.get_it()

        ele = reproject_data(self.global_dem, self.s2.s2_file_dir+'/%s.jp2'%target_band)
        ele.get_it()
        mask = ~np.isfinite(ele.data)
        ele.data[mask] = np.interp(np.flatnonzero(mask), \
                                   np.flatnonzero(~mask), ele.data[~mask]) # simple interpolation

        return aod.data, tcwv.data, tco3.data, ele.data


    def fire_correction(self, toa, sza, vza, saa, vaa, aod, tcwv, tco3, elevation, band_indexs):
        self._toa         = toa
        self._sza         = sza
        self._vza         = vza
        self._saa         = saa
        self._vaa         = vaa
        self._aod         = aod
        self._tcwv        = tcwv
        self._tco3        = tco3
        self._elevation   = elevation
        self._band_indexs = band_indexs
        self.corrected    = []
        rows              = np.repeat(np.arange(self.num_blocks), self.num_blocks)
        columns           = np.tile(np.arange(self.num_blocks), self.num_blocks)
        blocks            = zip(rows, columns)
        #self._s2_block_correction_6s(blocks[0])
        ret               = parmap(self._s2_block_correction_6s, blocks) 
        #ret               = parmap(self._s2_block_correction_emus, blocks) 
                
    def atm(self, p, RSR=None):
	aod, tcwv, tco3, sza, vza, raa , elevation = p
	path = '/home/ucfafyi/DATA/Multiply/6S/6SV2.1/sixsV2.1'
	s = SixS(path)
	s.altitudes.set_target_custom_altitude(elevation)
	s.altitudes.set_sensor_satellite_level()
	s.ground_reflectance = GroundReflectance.HomogeneousLambertian(GroundReflectance.GreenVegetation)
	s.geometry           = Geometry.User()
	s.geometry.solar_a   = 0
	s.geometry.solar_z   = sza
	s.geometry.view_a    = raa
	s.geometry.view_z    = vza
	s.aero_profile       = AeroProfile.PredefinedType(AeroProfile.Continental)
	s.aot550             = aod
	s.atmos_profile      = AtmosProfile.UserWaterAndOzone(tcwv, tco3)
	s.wavelength         = Wavelength(RSR)
	s.atmos_corr         = AtmosCorr.AtmosCorrLambertianFromReflectance(0.2)
	s.run()
	return s.outputs.coef_xap, s.outputs.coef_xbp, s.outputs.coef_xcp

    def _s2_block_correction_6s(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i,j))
        slice_x   = slice(i*self.block_size,(i+1)*self.block_size, 1)
        slice_y   = slice(j*self.block_size,(j+1)*self.block_size, 1)

        toa       = self._toa      [:,slice_x,slice_y]
        vza       = self._vza      [:,slice_x,slice_y]
        vaa       = self._vaa      [:,slice_x,slice_y]
        sza       = self._sza      [slice_x,slice_y]
        saa       = self._saa      [slice_x,slice_y]
        tcwv      = self._tcwv     [slice_x,slice_y]
        tco3      = self._tco3     [slice_x,slice_y]
        aod       = self._aod      [slice_x,slice_y]
        elevation = self._elevation[slice_x,slice_y]/1000.
        corfs = []
        for bi, rsr in enumerate(self.rsr):
            p     = np.nanmean(aod), np.nanmean(tcwv), np.nanmean(tco3), np.nanmean(sza), \
                    np.nanmean(vza[bi]), np.nanmean([saa - vaa[bi]]), np.nanmean(elevation) 
            a,b,c = self.atm(p, rsr)
            y     = a * toa[bi] -b
            corf  = y / (1 + c*y)
            corfs.append(corf)
        boa = np.array(corfs)
        self.corrected.append([i, j, boa])
        
    def _s2_block_correction_emus(self, block):
        i, j      = block
        self.logger.info('Block %03d--%03d'%(i,j))
        slice_x   = slice(i*self.block_size,(i+1)*self.block_size, 1)
        slice_y   = slice(j*self.block_size,(j+1)*self.block_size, 1)

        toa       =      self._toa[:,slice_x,slice_y].reshape(self._toa.shape[0], -1)
        vza       = list(self._vza[:,slice_x,slice_y].reshape(self._vza.shape[0], -1)*np.pi/180.)
        vaa       = list(self._vaa[:,slice_x,slice_y].reshape(self._vaa.shape[0], -1))
        
        sza       = self._sza      [slice_x,slice_y].ravel()*np.pi/180.
        saa       = self._saa      [slice_x,slice_y].ravel()
        tcwv      = self._tcwv     [slice_x,slice_y].ravel()
        tco3      = self._tco3     [slice_x,slice_y].ravel()
        aod       = self._aod      [slice_x,slice_y].ravel()
        elevation = self._elevation[slice_x,slice_y].ravel()/1000.

        boa       = self.correction_engine(toa, sza, vza, saa, vaa, aod, tcwv, tco3, elevation, self._band_indexs)
        
        self.corrected.append([i, j, boa])
        
    def correction_engine(self, toa, sza, vza, saa, vaa, aod, tcwv, tco3, elevation, band_indexs):
        atmos = np.array([aod, tcwv, tco3])
        boa,_ = self.s2_inv_AEE.emulator_reflectance_atmosphere(toa, atmos, sza,vza, \
                                                                 saa, vaa, elevation, bands=band_indexs)
        return np.array(boa)   

if __name__=='__main__':
    atmo_cor = atmospheric_correction(2017, 9, 4, '29SQB')
    atmo_cor.atmospheric_correction()
