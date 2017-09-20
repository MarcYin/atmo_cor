import gdal
import numpy as np
import numpy.ma as ma
import kernels

def r_modis(fname, slic=None):
    g = gdal.Open(fname)
    if g is None:
        raise IOError
    else:
        if slic==None:
            return g.ReadAsArray()
        elif g.RasterCount==1: 
            Lx,Ly = slic
            return g.ReadAsArray()[Lx,Ly]
        elif g.RasterCount>1:
            Lx,Ly = slic
            return g.ReadAsArray()[:, Lx, Ly]
        else:
            raise IOError

#bands = [2,3,4,8,13,11,12]


def get_kk(angles):
    vza ,sza,raa = angles
    kk = kernels.Kernels(vza ,sza,raa,\
                         RossHS=False,MODISSPARSE=True,\
                         RecipFlag=True,normalise=1,\
                         doIntegrals=False,LiType='Sparse',RossType='Thick')
    return kk


def qa_to_ReW(modisQAs, bands):
    magic = 0.618034
    modis = r_modis(modisQAs[3][0])
    QA = np.array([np.right_shift(np.bitwise_and(modis, np.left_shift(15,i*4)), i*4) for i in np.arange(0,7)])[bands,]
    relative_W = magic**QA
    relative_W[relative_W<magic**4]=0
    return relative_W

def get_rs(modisQAs, modis_filenames, angles, bands = range(7)):
    
    kk = get_kk(angles)
    k_vol = kk.Ross
    k_geo = kk.Li

    br = np.array([r_modis(modis_filenames[i][0]) for i in bands])
    mask = (br[:,0,:,:] > 32766) | (br[:,1,:,:] > 32766) |(br[:,2,:,:] > 32766)
    rw = qa_to_ReW(modisQAs, bands) # correpsonding relative weights
    brdf = br[:,0,:,:] + (br[:,1,:,:].T*k_vol).T + (br[:,2,:,:].T*k_geo).T
    brdf = ma.array(brdf, mask = mask)

    return [brdf,rw]


def get_brdf_six(fname, angles, bands = (7,), flag=None, Linds = None):    
    temp1 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'
    temp2 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'
    
    kk = get_kk(angles)
    k_vol = kk.Ross
    k_geo = kk.Li
    if Linds==None:
        br = np.array([r_modis(temp1%(fname, band)) for band in bands])
        qa = np.array([r_modis(temp2%(fname, band)) for band in bands])
        #mask = (br[:,0,:,:] > 32766) | (br[:,1,:,:] > 32766) |(br[:,2,:,:] > 32766)
        brdf = br[:,0,:,:] + br[:,1,:,:]*k_vol + br[:,2,:,:]*k_geo
        #brdf = ma.array(brdf, mask = mask)
        return [brdf*0.001, qa]
    else:
        Lx, Ly = Linds
        br = np.array([r_modis(temp1%(fname, band), slic=[Lx, Ly]) for band in bands])
        qa = np.array([r_modis(temp2%(fname, band), slic=[Lx, Ly]) for band in bands])
        brdf = br[:,0] + br[:,1]*k_vol + br[:,2]*k_geo
        if flag==None:
            return [brdf*0.001, qa]
        else:
            mask = (qa<=flag)
            #val_brdf = brdf[:,mask]
            #val_ins = np.array(Linds)[:,mask]
            return [brdf*0.001, mask]
