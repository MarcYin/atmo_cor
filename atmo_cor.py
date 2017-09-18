#/usr/bin/env python
import numpy as np
import sys
sys.path.insert(0, 'python')
from multiprocessing import pool
import cPickle as pkl
from functools import partial
from emulation_engine import AtmosphericEmulationEngine
from grab_uncertainty import grab_uncertainty
class atmo_cor(object):
    '''
    A class taking the toa, boa, initial [aot, water, ozone], [vza, sza, vaa, saa], elevation and emulators
    to do the atmospheric parameters retrival and do the atmopsheric correction.  
    '''
    def __init__(self, sensor,
                       emus_dir,
                       boa,toa, 
                       atmosphere, 
                       sza, vza,
                       saa, vaa, 
                       elevation,
                       boa_qa, boa_bands,
                       band_indexs, 
                       mask,prior,
                       subsample = None,
                       subsample_start = 0,
                       gradient_refl=True, 
                       bands=None):
        
        self.alpha         = 1.42 #angstrom exponent for continental type aerosols
        self.sensor        = sensor
        self.emus_dir      = emus_dir
        self.boa, self.toa = boa, toa
        self.atmosphere    = atmosphere
        self.sza, self.vza = sza, vza
        self.saa, self.vaa = saa, vaa
        self.elevation     = elevation
        self.boa_qa        = boa_qa
        self.boa_bands     = boa_bands
        self.band_weights  = (np.array(self.boa_bands)/1000.)**self.alpha
        self.band_indexs   =  band_indexs
        self.mask          = mask
        self.prior         = prior
        if subsample is None:
            self.subsample = 1
        else:
           self.subsample  = subsample
        self.subsample_sta = subsample_start
    
    def _load_emus(self):
        self.AEE = AtmosphericEmulationEngine(self.sensor, self.emus_dir)
    def _load_unc(self):
        uc = grab_uncertainty(self.boa, self.boa_bands, self.boa_qa )
        self.boa_unc   = uc.get_boa_unc()
        self.aot_unc   = uc.aot_unc
        self.water_unc = uc.water_unc
        self.ozone_unc = uc.ozone_unc

    def _sort_emus_inputs(self,):

        assert self.boa.shape[-2:] == self.mask.shape, 'mask should have the same shape as the last two axises of boa.'
        assert self.boa.shape      == self.toa.shape, 'toa and boa should have the same shape.'
        assert self.boa.shape      == self.boa_unc.shape, 'boa and boa_unc should have the same shape.'
        assert self.atmosphere.shape[0] == 3, 'Three parameters, i.e. AOT, water and Ozone are needed.'
        assert self.boa.shape[-2:] == self.atmosphere.shape[-2:], 'boa and atmosphere should have the same shape in the last two axises.'
        # make the boa and toa to be the shape of nbands * nsample
        # and apply the flattened mask and subsample 
        flat_mask    = self.mask.flatten()
        flat_boa     = self.boa.reshape(self.boa.shape[0], -1)[...,flat_mask][...,self.subsample_sta::self.subsample]
        flat_toa     = self.toa.reshape(self.toa.shape[0], -1)[...,flat_mask][...,self.subsample_sta::self.subsample]
        flat_boa_unc = self.boa_unc.reshape(self.toa.shape[0], -1)[...,flat_mask][...,self.subsample_sta::self.subsample]
        flat_atmos   = self.atmosphere.reshape(3, -1)[...,flat_mask][...,self.subsample_sta::self.subsample]
        flat_angs_ele = []
        for i in [self.sza, self.vza, self.saa, self.vaa, self.elevation]:
            if isinstance(i, (float,int)):
                flat_angs_ele.append(i)
            else:
                assert i.shape == self.boa.shape[-2:], 'i should have the same shape as the last two axises of boa.'
                flat_i = i.flatten()[flat_mask][self.subsample_sta::self.subsample]
                flat_angs_ele.append(flat_i)
        ## for the prior
        if np.array(self.prior).ndim == 1:
            self.flat_prior = np.array(self.prior) 
        else:
            assert self.prior.shape == self.boa.shape[-2:], 'prior should have the same shape as the last two axises of boa.'
            self.flat_prior = self.prior.reshape(3, -1)[...,flat_mask][...,self.subsample_sta::self.subsample]
        self.flat_atmos = flat_atmos

        return flat_mask, flat_boa, flat_toa, flat_boa_unc, flat_atmos, flat_angs_ele # [sza, vza, saa, vaa, elevation]        

    def obs_cost(self,):

        flat_mask, flat_boa, flat_toa, flat_boa_unc, flat_atmos, [sza, vza, saa, vaa, elevation] = self._sort_emus_inputs()
        H0, dH = self.AEE.emulator_reflectance_atmosphere(flat_boa, flat_atmos, sza, vza, saa, vaa, elevation, bands=self.band_indexs)
        H0, dH = np.array(H0), np.array(dH)
        diff = (flat_toa - H0) #[..., None] dd an extra dimentiaon to match the dH 
        correction_mask = np.isfinite(diff).all(axis=0)
        diff[:,~correction_mask] = 0.
        dH[:,~correction_mask,:] = 0.
        J  = (0.5 * self.band_weights[...,None] * diff**2 / flat_boa_unc**2).sum(axis=(0,1))
        full_dJ = [ self.band_weights[...,None] * dH[:,:,i] * diff/flat_boa_unc**2 for i in xrange(4,7)]
        J_ = np.array(full_dJ).sum(axis=(1,2))
        
        return J, J_

    def prior_cost(self,):
        J = [0.5 * (self.flat_atmos - self.flat_prior[...,None])**2/unc**2 for unc in (self.aot_unc, self.water_unc, self.ozone_unc)]
        full_dJ = [(self.flat_atmos - self.flat_prior[...,None])/unc**2 for unc in (self.aot_unc, self.water_unc, self.ozone_unc)]
        J_ = np.array(full_dJ).sum(axis=(1,2))
        J  = np.array(J).sum()
        return J, J_
    def smooth_cost(self,):
        '''
        need to add first order regulization
        '''
        J  = 0
        J_ = np.array([0,0,0])
        return J, J_


if __name__ == "__main__":
    boa = np.random.rand(4,100,100)
    boa[:] = 0.2
    toa = np.random.rand(4,100,100)
    toa[:] = 0.3
    aot = np.random.rand(100, 100)
    aot[:] = 0.3
    water = np.random.rand(100, 100)
    water[:] = 3.4
    ozone = np.zeros((100, 100))
    ozone[:] = 0.35
    atmosphere = np.array([aot,water,ozone])
    boa_qa = np.random.choice([0,1,255], size=(4,100,100))
    mask,prior = np.zeros((100, 100)).astype(bool), [0.2, 3, 0.3]
    mask[:50,:50] = True
    atom = atmo_cor('MSI', '/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators',boa, toa,atmosphere,0.5,0.5,10,10,0.5, boa_qa, boa_bands=[645,869,469,555], band_indexs=[3,7,1,2], mask=mask, prior=prior)
    atom._load_emus()
    atom._load_unc()
    atom._sort_emus_inputs()
    obs_J, obs_J_ = atom.obs_cost()
    prior_J, prior_J_ = atom.prior_cost()
    smooth_J, smooth_J_ = atom.smooth_cost()
    J = obs_J + prior_J + smooth_J
    J_ = obs_J_ +  prior_J_ + smooth_J_
