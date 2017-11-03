#/usr/bin/env python 
import sys
sys.path.insert(0, 'python')
import numpy as np
from glob import glob
import cPickle as pkl
from multi_process import parmap

class solving_atmo_paras(object): 
    '''
    A simple implementation of dark dense vegitation method for the restieval of prior aod.
    '''
    def __init__(self,
                 boa, toa,
                 sza, vza,
                 saa, vaa,
                 aod_prior,
                 tcwv_prior,
                 tco3_prior,
                 elevation,
                 aod_unc,
                 tcwv_unc,
                 tco3_unc,
                 boa_unc,
                 Hx, Hy,
                 full_res,
                 emulators, 
                 band_indexs,
                 band_wavelength,
                 alpha = -1.42
                 subsample = 1
                 subsample_start = 0
                 ):
        
        self.boa             = boa
        self.toa             = toa
        self.sza             = np.cos(sza*np.pi/180.)
        self.vza             = np.cos(vza*np.pi/180.)
        self.saa             = np.cos(saa*np.pi/180.)
        self.vaa             = np.cos(vaa*np.pi/180.)
        self.raa             = np.cos((self.saa - self.vaa)*np.pi/180.)
        self.aod_prior       = aod_prior
        self.tcwv_prior      = self.tcwv_prior
        self.tco3_prior      = self.tco3_prior
        self.ele             = elevation
        self.aod_unc         = aod_unc
        self.tcwv_unc        = tcwv_unc
        self.tco3_unc        = tco3_unc
        self.boa_unc         = boa_unc
        self.Hx, self.Hy     = Hx, Hy
        self.full_res        = full_res
        self.emus            = emulators
        self.band_indexs     = band_indexs
        self.alpha           = alpha
        self.band_weights    = (np.array(self.band_wavelength)/1000.)**self.alpha
        self.band_weights    = self.band_weights / self.band_weights.sum() 
        self.subsample       = subsample
        self.subsample_start = subsample_start
 
    def _pre_process(self,):
        try:
            zero_aod  = np.zeros_loke(self.sza)
            zero_tcwv = np.zeros_loke(self.sza)
            zero_tco3 = np.zeros_loke(self.sza)
            self.control_variables = np.zeros((self.boa.shape[0], 7) + self.vza.shape)
            if self.vza.ndim == 2:
                self.control_variables[:] = np.array([self.sza, self.vza, self.raa, zero_aod, zero_tcwv, zero_tco3, self.ele])
            elif self.vza.ndim == 3:
                assert self.vza.shape[0] == self.boa.shape[0], 'Each band should have corresponding angles.'
                for i in range(len(self.vza)):
                    self.control_variables[i] = np.array([self.sza, self.vza[i], self.raa[i], zero_aod, zero_tcwv, zero_tco3, self.ele])
            else:
                raise IOError('Angles should be a 2D array.')
        except:
            raise IOError('Check the shape of input angles and elevation.') 
       
        try:
            self.uncs = np.array([self.aod_unc, self.tcwv_unc, self.tco3_unc, self.boa_unc])
        except:
            raise IOError('Check the shape of input uncertainties.')
        self.resample_hx = (1. * self.Hx / self.full_res[0] * self.vza.shape[0]).astype(int)
        self.resample_hy = (1. * self.Hy / self.full_res[1] * self.vza.shape[1]).astype(int)
        self.xap_emus    = self.emus[0][self.band_indexs]
        self.xbp_emus    = self.emus[1][self.band_indexs]
        self.xcp_emus    = self.emus[2][self.band_indexs]
        self.uncs        = self.uncs[:, self.resample_hx, self.resample_hx]
    def _ddv_prior(self,):
        
        self._load_xa_xb_xc_emus()
        ndvi = (self.nir - self.red)/(self.nir + self.red)
        ndvi_mask = (ndvi > 0.6) & (self.swif > 0.01) & (self.swif < 0.25)
        if ndvi_mask.sum() < 3:
            return (-9999, 9999) # need to have at least 100 pixels to get a relative good estimation of aod
        elif ndvi_mask.sum() > 25000000:
            Hx, Hy                      = np.where(ndvi_mask)
            random_choice               = np.random.choice(len(Hx), 25000000, replace=False)
            random_choice.sort()
            self.Hx, self.Hy            = Hx[random_choice], Hy[random_choice]
            new_mask                    = np.zeros_like(self.blue).astype(bool)
            new_mask[self.Hx, self.Hy]  = True
            self._ndvi_mask             = new_mask
        else:
            self.Hx, self.Hy            = np.where(ndvi_mask)
            self._ndvi_mask             = ndvi_mask

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
        self.resample_hx                = (1. * self.Hx / self.blue.shape[0] * self.num_blocks).astype(int)
        self.resample_hy                = (1. * self.Hy / self.blue.shape[1] * self.num_blocks).astype(int)
        solved                          = self._optimization()
        return solved

    def _block_resample(self, parameter):
        hx = np.repeat(range(self.num_blocks), self.num_blocks)
        hy = np.tile  (range(self.num_blocks), self.num_blocks)
        x_size, y_size = parameter.shape
        resample_x = (1.* hx / self.num_blocks*x_size).astype(int)
        resample_y = (1.* hy / self.num_blocks*y_size).astype(int)
        resampled_parameter = parameter[resample_x, resample_y].reshape(self.num_blocks, self.num_blocks)
        return resampled_parameter

    def _bos_cost(self, p, is_full = True):

        X             = self.control_variables.reshape(self.boa.shape[0], 7, -1)
        X[:, 3:6, :]  = np.array(p)
        xap_H,  xbp_H,  xcp_H  = [], [], []
        xap_dH, xbp_dH, xcp_dH = [], [], []
          
        for i in range(len(self.xap_emus)):
            H, dH   = self.xap_emus[i].predict(X[i].T, do_unc=False) 
            H, dH   = np.array(H).reshape(*self.sza.shape), np.array(dH)[:,3:6].reshape(*self.sza.shape, 3)
            xap_H. append(H [self.resample_hx,self.resample_hy])
            xap_dH.append(dH[self.resample_hx,self.resample_hy,:])

            H, dH   = self.xbp_emus[i].predict(X[i].T, do_unc=False) 
            H, dH   = np.array(H).reshape(*self.sza.shape), np.array(dH)[:,3:6].reshape(*self.sza.shape, 3)
            xbp_H. append(H [self.resample_hx,self.resample_hy])
            xbp_dH.append(dH[self.resample_hx,self.resample_hy,:])

            H, dH   = self.xcp_emus[i].predict(X[i].T, do_unc=False) 
            H, dH   = np.array(H).reshape(*self.sza.shape), np.array(dH)[:,3:6].reshape(*self.sza.shape, 3)
            xcp_H. append(H [self.resample_hx,self.resample_hy])
            xcp_dH.append(dH[self.resample_hx,self.resample_hy,:])

        xap_H,  xbp_H,  xcp_H  = np.array(xap_H),    np.array(xbp_H),    np.array(xcp_H)
        xap_dH, xbp_dH, xcp_dH = np.array(xap_dH), np.array(xbp_dH), np.array(xcp_dH)
        
        y        = xap_H * self.toa - xbp_H
        sur_ref  = y / (1 + xcp_H * y) 
        diff     = sur_ref - self.boa
        J        = (0.5 * self.band_weights[...,None] * (diff)**2 / self.uncs[3]).sum()
        dH       = (-self.toa[...,None] * xap_dH + xcp_dH (xbp_H - xap_H * self.toa[...,None])**2 + \
                    xbp_dH) /(self.toa[...,None] * xap_H * xcp_H - xbp_H * xcp_H + 1)**2
        full_dJ  = [ self.band_weights[...,None] * dH[:,:,i] * diff / (self.uncs[3]**2) for i in range(3)]
        if is_full:
            J_ = np.array(full_dJ).sum(axis=(1,))
        else:
            J_ = np.array(full_dJ).sum(axis=(1, 2))
        return J, J_
         
    def _smooth_cost(self, aod):
        aod   = aod.reshape(self.num_blocks, self.num_blocks)
        #s     = smoothn(aod, isrobust=True, verbose=False)[1]
        smed  = smoothn(aod, isrobust=True, verbose=False, s = 1)[0]
        cost = (0.5 * (smed - aod)**2)[self.resample_hx, self.resample_hy]
        return cost
 
    def _cost(self, aod):
        J_obs = self._bos_cost(aod)
        #J_smo = self._smooth_cost(aod)
        #print 'smooth cost: ',J_smo.sum()
        #print 'obs cost: ', J_obs.sum()
        #print aod, J_obs.sum()
        return J_obs.sum() #+ J_smo.sum()

        
    def _optimization(self,):
        #p0      = np.zeros((self.num_blocks, self.num_blocks)).ravel()
        #bot     = np.zeros((self.num_blocks, self.num_blocks)).ravel()
        #up      = np.zeros((self.num_blocks, self.num_blocks)).ravel()
        #up[:]   = 2
        #bounds  = np.array([bot, up]).T 
        #p0[:]   = 0.3
        p       = np.r_[np.arange(0, 1., 0.02), np.arange(1., 1.5, 0.05),  np.arange(1.5, 2., 0.1)]
        costs   = parmap(self._cost, p)
        min_ind = np.argmin(costs) 
        return p[min_ind], costs[min_ind]
        #psolve = optimize.fmin_l_bfgs_b(self._cost, p0, approx_grad = 1, iprint = 1, maxiter= 3,\
        #                                pgtol = 1e-4,factr=1000, bounds = bounds,fprime=None)
        #return psolve
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
    b2  = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2016/2/3/0/B02.jp2').ReadAsArray()/10000.
    b4  = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2016/2/3/0/B04.jp2').ReadAsArray()/10000.
    b8  = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2016/2/3/0/B08.jp2').ReadAsArray()/10000.
    b12 = gdal.Open('/home/ucfafyi/DATA/S2_MODIS/s_data/50/S/LG/2016/2/3/0/B12.jp2').ReadAsArray()/10000.
    b12 = np.repeat(np.repeat(b12, 2, axis = 1), 2, axis = 0)
    this_ddv = ddv(b2, b4, b8, b12, 'MSI', sza, vza, raa, ele, tcwv, tco3, band_index = [1, 3])
    
    solved = this_ddv._ddv_prior()
    #solevd = this_ddv._optimization()
