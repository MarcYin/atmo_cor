#/usr/bin/env python
import sys
sys.path.insert(0, 'python')
from glob import glob
from get_modis_toa import grab_modis_toa
import cPickle as pkl
from get_brdf import get_brdf_six
import numpy as np
from atmo_cor import atmo_cor

class prepare_modis(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,h,v,
                 year, doy,
                 mcd43_dir   = '/home/ucfafyi/DATA/S2_MODIS/m_data/',
                 mod_l1b_dir = '/data/selene/ucfajlg/Bondville_MODIS/THERMAL',
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 l8_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/l_data/',
                 qa_thresh   = 255,
                 mod_cloud   = None,
                 verbose     = True,
                 save_file   = False):

        self.year        = year 
        self.doy         = doy
        self.h           = h
        self.v           = v
        self.mcd43_dir   = mcd43_dir
        self.mod_l1b_dir = mod_l1b_dir
        self.qa_thresh   = qa_thresh
        self.mod_cloud   = mod_cloud 
        self.s2_toa_dir  = s2_toa_dir
        self.l8_toa_dir  = l8_toa_dir


    def _locate_files(self, save_file=False):
        mcd43_tmp       = '%s/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'
        self.mcd43_file = glob(mcd43_tmp%(self.mcd43_dir,\
                                self.year, self.doy, self.h, self.v))[0]

        self.modis_toa, self.modis_angles = grab_modis_toa(year=2006,doy=200,verbose=True,\
                                                           mcd43file = self.mcd43_file, directory_l1b= self.mod_l1b_dir)
        for i, modis_toa in enumerate( self.modis_toa):
            modis_angle = self.modis_angles[i]
            self.modis_boa, self.qa = get_brdf_six(self.mcd43_file, (modis_angle[0],\
                                                   modis_angle[1], modis_angle[2] - modis_angle[3]),\
                                                   bands=[1,2,3,4,5,6,7], flag=None, Linds= None)
            if self.mod_cloud is None:
                self.modis_cloud = np.zeros_like(modis_toa[0]).astype(bool)

            self.quality_mask = np.all(self.qa <= self.qa_thresh, axis =0) 
            self.valid_mask   = np.all(~self.modis_boa.mask, axis = 0)
            self.toa_mask     = np.all(np.isfinite(modis_toa), axis=0)
            self.mask         = self.quality_mask & self.valid_mask & self.toa_mask & (~self.modis_cloud)
            self.patch_mask   = np.zeros_like(self.mask).astype(bool)
            self.patch_mask[100:200,100:200] = True 

            boa, toa  = self.modis_boa[:,self.patch_mask].reshape(7,100, 100), modis_toa[:,self.patch_mask].reshape(7,100, 100)
            vza, sza  = np.cos(modis_angle[:2, self.patch_mask]).reshape(2,100, 100)
            vaa, saa  =        modis_angle[2:, self.patch_mask].reshape(2,100, 100)
            boa_qa    = self.qa[:,self.patch_mask].reshape(7,100,100)
            mask      = self.mask[self.patch_mask].reshape(100, 100)
            prior     = 0.2, 3.4, 0.35
            aot       = np.zeros((100, 100)) 
            water     = aot.copy()
            ozone     =  aot.copy()
            aot[:]=0.2; water[:] = 3.4; ozone[:] = 0.35
            atmosphere= np.array([aot, water, ozone])
            self.atom = atmo_cor('TERRA', '/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators',boa, \
                        toa,sza,vza,saa,vaa,0.5, boa_qa, boa_bands=[645,869,469,555,1240,1640,2130], \
                        band_indexs=[0,1,2,3,4,5,6], mask=mask, prior=prior, atmosphere = atmosphere, subsample=10)

            self.atom._load_unc()
            if i==0:
                self.atom._load_emus()
                self.AEE    = self.atom.AEE
                self.bounds = self.atom.bounds
            else:
                self.atom.AEE    = self.AEE
                self.atom.bounds =  self.bounds
            if mask.sum() == 0:
                pass
            else:
                print self.atom.optimization()

    #def S2_aerosol(self,):
        


     
if __name__ == "__main__":
    pre_mod = prepare_modis(11,4,2006, 200)
    pre_mod._locate_files() 
