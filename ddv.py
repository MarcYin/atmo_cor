#/usr/bin/env python 
import sys
sys.path.insert(0,'python')
import numpy as np
from glob import glob
import cPickle as pkl
from smoothn import smoothn
from scipy import optimize 
from multi_process import parmap

class ddv_aod(object):
    '''
    A simple implementation of dark dense vegitation method for the restieval of prior aod.
    '''
    def __init__(self,
                 blue, red,
                 nir, swif,
                 sensor,
                 sza, vza, raa,
                 elevation,
                 tcwv, tco3,
                 band_index,
                 block_size = 180,
                 Hx = None, 
                 Hy = None,
                 emus_dir = '/home/ucfafyi/DATA/Multiply/emus/'
                 ):
        self.blue    = blue
        self.red     = red
        self.nir     = nir
        self.swif    = swif
        self.sensor  = sensor
        self.sza     = np.cos(sza*np.pi/180.)
        self.vza     = np.cos(vza*np.pi/180.)
        self.raa     = np.cos(raa*np.pi/180.)
        self.ele     = elevation
        self.tcwv    = tcwv
        self.tco3    = tco3
        self.blue_in = band_index[0]
        self.red_in  = band_index[1]
        if (Hx is not None) & (Hy is not None):
            self.Hx = Hx
            self.Hy = Hy
        else:
            self.Hx = np.repeat(range(self.blue.shape[0]), self.blue.shape[1])
            self.Hy = np.tile  (range(self.blue.shape[1]), self.blue.shape[0])

        self.block_size = block_size
        self.emus_dir   = emus_dir
    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap.pkl'%(self.sensor))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp.pkl'%(self.sensor))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp.pkl'%(self.sensor))[0]
        f = lambda em: pkl.load(open(em, 'rb'))
        self.xap_emus, self.xbp_emus, self.xcp_emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def _ddv_prior(self,):
        self._load_xa_xb_xc_emus()
        ndvi = (self.nir - self.red)/(self.nir + self.red)
        self.blue_xap_emu, self.blue_xbp_emu, self.blue_xcp_emu = self.xap_emus[self.blue_in], \
                                                                  self.xbp_emus[self.blue_in], self.xcp_emus[self.blue_in]
        self.red_xap_emu,  self.red_xbp_emu,  self.red_xcp_emu  = self.xap_emus[self.red_in],  \
                                                                  self.xbp_emus[self.red_in],  self.xcp_emus[self.red_in]
      
        self._ndvi_mask = (ndvi > 0.6) & (self.swif > 0.01) & (self.swif < 0.25)
        self.num_blocks = self.blue.shape[0] / self.block_size
        zero_aod = np.zeros((self.num_blocks, self.num_blocks))
        if self.vza.ndim == 3:
            blue_resampled_parameters = []
            for parameter in [self.sza, self.vza[0], self.raa[0], zero_aod, self.tcwv, self.tco3, self.ele]:
                blue_resampled_parameters.append(self._block_resample(parameter).ravel())
            blue_resampled_parameters   = np.array(blue_resampled_parameters)
            red_resampled_parameters    = np.array(blue_resampled_parameters).copy()
            red_resampled_parameters[1] = self._block_resample(self.vza[1]).ravel()
            red_resampled_parameters[2] = self._block_resample(self.raa[1]).ravel() 
        elif self.vza.ndim == 2:
            blue_resampled_parameters = []
            for parameter in [self.sza, self.vza, self.raa, zero_aod, self.tcwv, self.tco3, self.ele]:
                blue_resampled_parameters.append(self._block_resample(parameter).ravel())
            blue_resampled_parameters   = np.array(blue_resampled_parameters)
            red_resampled_parameters    = np.array(blue_resampled_parameters).copy()
        else:
            raise IOError('Angles should be 2D array or several 2D array (3D)...')
        self.blue_resampled_parameters  = blue_resampled_parameters
        self.red_resampled_parameters   = red_resampled_parameters  
        self.Hx, self.Hy                = np.where(self._ndvi_mask)
        self.resample_hx                = (1. * self.Hx / self.blue.shape[0] * self.num_blocks).astype(int)
        self.resample_hy                = (1. * self.Hy / self.blue.shape[1] * self.num_blocks).astype(int)

    def _block_resample(self, parameter):
        hx = np.repeat(range(self.num_blocks), self.num_blocks)
        hy = np.tile  (range(self.num_blocks), self.num_blocks)
        x_size, y_size = parameter.shape
        resample_x = (1.* hx / self.num_blocks*x_size).astype(int)
        resample_y = (1.* hy / self.num_blocks*y_size).astype(int)
        resampled_parameter = parameter[resample_x, resample_y].reshape(self.num_blocks, self.num_blocks)
        return resampled_parameter

    def _bos_cost(self, aod):
        self.blue_resampled_parameters[3] = aod
        self.red_resampled_parameters[3]  = aod
        blue_xap, blue_xbp, blue_xcp      = self.blue_xap_emu.predict(self.blue_resampled_parameters.T)[0].reshape(self.num_blocks, self.num_blocks)[self.resample_hx, self.resample_hy],\
                                            self.blue_xbp_emu.predict(self.blue_resampled_parameters.T)[0].reshape(self.num_blocks, self.num_blocks)[self.resample_hx, self.resample_hy],\
                                            self.blue_xcp_emu.predict(self.blue_resampled_parameters.T)[0].reshape(self.num_blocks, self.num_blocks)[self.resample_hx, self.resample_hy]

        red_xap, red_xbp, red_xcp         = self.red_xap_emu.predict(self.red_resampled_parameters.T)[0].reshape(self.num_blocks, self.num_blocks)[self.resample_hx, self.resample_hy],\
                                            self.red_xbp_emu.predict(self.red_resampled_parameters.T)[0].reshape(self.num_blocks, self.num_blocks)[self.resample_hx, self.resample_hy],\
                                            self.red_xcp_emu.predict(self.red_resampled_parameters.T)[0].reshape(self.num_blocks, self.num_blocks)[self.resample_hx, self.resample_hy] 
       
        y        = blue_xap * self.blue[self._ndvi_mask] - blue_xbp
        blue_sur = y / (1 + blue_xcp * y) 
        y        = red_xap * self.red[self._ndvi_mask] - red_xbp
        red_sur  = y / (1 + red_xcp * y)
        blue_dif = (blue_sur - 0.25 * self.swif[self._ndvi_mask])**2
        red_dif  = (blue_sur - 0.5  * self.swif[self._ndvi_mask])**2
        cost     = 0.5 * (blue_dif + red_dif)
        return cost
         
    def _smooth_cost(self, aod):
        aod   = aod.reshape(self.num_blocks, self.num_blocks)
        #s     = smoothn(aod, isrobust=True, verbose=False)[1]
        smed  = smoothn(aod, isrobust=True, verbose=False, s = 1)[0]
        cost = 0.5 * (smed - aod)**2
        return cost
 
    def _cost(self, aod):
        J_obs = self._bos_cost(aod)
        J_smo = self._smooth_cost(aod)
        
        print 'smooth cost: ',J_smo.sum()
        print 'obs cost: ', J_obs.sum()
        return J_obs.sum() #+ J_smo.sum()

        
    def _optimization(self,):
        p0    = np.zeros((self.num_blocks, self.num_blocks)).ravel()
        bot = np.zeros((self.num_blocks, self.num_blocks)).ravel()
        up  = np.zeros((self.num_blocks, self.num_blocks)).ravel()
        up[:] = 2
        bounds = np.array([bot, up]).T 
        p0[:] = 0.2 
        psolve = optimize.fmin_l_bfgs_b(self._cost, p0, approx_grad=1, iprint = 1, \
                                        pgtol=1e-6,factr=1000, bounds=bounds,fprime=None)
        return psolve
if __name__ == '__main__':
    import gdal
    sza  = np.ones((23,23))
    vza  = np.ones((23,23))
    raa  = np.ones((23,23))
    ele  = np.ones((61,61))
    tcwv = np.ones((61,61))
    tco3 = np.ones((61,61)) 
    sza[:]  = 30.
    vza[:]  = 10.
    raa[:]  = 100.
    ele[:]  = 0.02
    tcwv[:] = 2.3
    tco3[:] = 0.3
    b2  = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2017/9/22/0/B02.jp2').ReadAsArray()/10000.
    b4  = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2017/9/22/0/B04.jp2').ReadAsArray()/10000.
    b8  = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2017/9/22/0/B08.jp2').ReadAsArray()/10000.
    b12 = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2017/9/22/0/B12.jp2').ReadAsArray()/10000.
    b12 = np.repeat(np.repeat(b12, 2, axis = 1), 2, axis = 0)
    this_ddv = ddv_aod(b2, b4, b8, b12, 'MSI', sza, vza, raa, ele, tcwv, tco3, band_index = [1, 3])
    
    this_ddv._ddv_prior()
    this_ddv._optimization()
