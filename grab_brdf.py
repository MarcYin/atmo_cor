import sys
sys.path.insert(0, 'python')
import gdal
import glob
import kernels
import numpy as np
from osgeo import osr
from functools import partial
from multi_process import parmap
from reproject import reproject_data
from datetime import datetime, timedelta

x_step = -463.31271653
y_step = 463.31271653
m_y0, m_x0 = -20015109.354, 10007554.677

 mod09_band = [1, 2, 3, 4, 5, 6, 7]
 mod09_band_unc = 0.0085, 0.0246, 0.0055, 0.0085, 0.0179, 0.0125, 0.0087
 mod09_band_unc_dict = dict(zip(mod09_band, mod09_band_unc))

def r_modis(fname, xoff = None, yoff = None, xsize = None, ysize = None):
    g = gdal.Open(fname)
    if g is None:
        raise IOError
    else:
        if x_off is None:
            return g.ReadAsArray()
        elif g.RasterCount==1:
            return g.ReadAsArray(xoff, yoff, xsize, ysize)
        elif g.RasterCount>1:
            for band in range(g.RasterCount):
                band += 1
                rets.append(g.GetRasterBand(band).ReadAsArray(xoff, yoff, xsize, ysize))
            return np.array(rets)
        else:
            raise IOError

def mtile_cal(lat, lon):
    # a function calculate the tile number for MODIS, based on the lat and lon
    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    modis_sinu = osr.SpatialReference() # define the SpatialReference object
    modis_sinu.ImportFromProj4 ( \
                    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    tx = osr.CoordinateTransformation( wgs84, modis_sinu)# from wgs84 to modis 
    ho,vo,z = tx.TransformPoint(lon, lat)# still use the function instead of using the equation....
    h = int((ho-m_y0)/(2400*y_step))
    v = int((vo-m_x0)/(2400*x_step))
    return h,v

def get_hv(example_file):
    g = gdal.Open(example_file)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterYSize, g.RasterXSize

    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    H_res_geo = osr.SpatialReference( )
    raster_wkt = g.GetProjection()
    H_res_geo.ImportFromWkt(raster_wkt)
    tx = osr.CoordinateTransformation(H_res_geo, wgs84)
    # so we need the four corners coordiates to check whether they are within the same modis tile
    (ul_lon, ul_lat, ulz ) = tx.TransformPoint( geo_t[0], geo_t[3])

    (lr_lon, lr_lat, lrz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
                                          geo_t[3] + geo_t[5]*y_size )

    (ll_lon, ll_lat, llz ) = tx.TransformPoint( geo_t[0] , \
                                          geo_t[3] + geo_t[5]*y_size )

    (ur_lon, ur_lat, urz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
                                          geo_t[3]  )
    a0, b0 = None, None
    corners = [(ul_lon, ul_lat), (lr_lon, lr_lat), (ll_lon, ll_lat), (ur_lon, ur_lat)]
    tiles = []
    for i,j  in enumerate(corners):
        h, v = mtile_cal(j[1], j[0])
        tiles.append('h%02dv%02d'%(h,v))
    unique_tile = np.unique(np.array(tiles))
    return unique_tile

def array_to_raster(array, example_file):
    if array.ndim == 2:
        bands = 1
    elif array.ndim ==3:
        bands = array.shape[0]
    else:
        raise IOError('Only 2 or 3 D array is supported.')
    try:
        g = gdal.Open(example_file)
    except:
        g = example_file
    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create('', array.shape[-1], array.shape[-2], bands, gdal.GDT_Float64)
    ds.SetProjection(g.GetProjection())
    geotransform    = list(g.GetGeoTransform())  
    geotransform[1] = geotransform[1] * g.RasterXSize / (1. * array.shape[-1])
    geotransform[5] = geotransform[5] * g.RasterYSize / (1. * array.shape[-2])
    ds.SetGeoTransform(geotransform)
    if array.ndim == 3:
        for i in range(bands):
            ds.GetRasterBand(i+1).WriteArray(array[i])
    else:
         ds.GetRasterBand(1).WriteArray(array)
    return ds

def get_kk(angles):
    vza ,sza,raa = angles
    kk = kernels.Kernels(vza ,sza,raa,\
                         RossHS=False,MODISSPARSE=True,\
                         RecipFlag=True,normalise=1,\
                         doIntegrals=False,LiType='Sparse',RossType='Thick')
    return kk

def get_brdf_six(MCD43_dir, example_file, year, doy, ang_files, sun_view_ang_scale=[1,1], bands = (7,), Linds = None, do_unc = True):
    f_temp = MCD43_dir + '/MCD43A1.A%s.%s.006*.hdf'
    temp1  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'
    temp2  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'
    
    unique_tile = get_hv(example_file)
    print unique_tile
    date   = datetime.strptime('%d%03d'%(year, doy), '%Y%j')
    days   = [(date - timedelta(days = i)).strftime('%Y%j') for i in np.arange(16, 0, -1)] + \
             [(date + timedelta(days = i)).strftime('%Y%j') for i in np.arange(0, 17,  1)]
    data_f = [[temp1%(glob.glob(f_temp%(day, tile))[0], band) for tile in unique_tile] for band in bands for day in days]  
    qa_f   = [[temp2%(glob.glob(f_temp%(day, tile))[0], band) for tile in unique_tile] for band in bands for day in days]
    g      = gdal.Open(example_file)
    temp_data = ~np.isnan(reproject_data(example_file, gdal.BuildVRT('', data_f[0]), outputType = gdal.GDT_Float64).data)
    print 'got vrt'
    max_x, max_y = np.array(np.where(temp_data)).max(axis=1)
    min_x, min_y = np.array(np.where(temp_data)).min(axis=1)
    xoff,  yoff  = min_y, min_x
    xsize, ysize = (max_y - min_y + 1), (max_x - min_x + 1)
    print 'read in data'
    f      = lambda fname: gdal.BuildVRT('', fname).ReadAsArray(xoff, yoff, xsize, ysize)
    data   = np.array(parmap(f, data_f)).reshape(len(bands), len(days), 3, ysize, xsize)
    data   = np.ma.array(data, mask = (data==32767)).astype(float)
    print 'read in qa'
    qa     = np.array(parmap(f, qa_f  )).reshape(len(bands), len(days),    ysize, xsize)
    w      = 0.618034 ** qa.astype(float)
    f      = lambda band: np.array(smoothn(data[band[0],:,band[1],:,:], s=10., smoothOrder=1., \
                                   axis=0, TolZ=0.001, verbose=True, isrobust=True, W = w[band[0]]))[[0,3],]
    ba     = np.array([np.tile(range(len(bands)), 3), np.repeat(range(3), len(bands))]).T
    print 'smoothing....'
    smed   = np.array(parmap(f, ba))
    dat    = np.concatenate(smed[:,0], axis=0).reshape(3, len(bands), len(days), ysize, \
                            xsize)[:,:,16, np.where(temp_data)[0]-min_x, np.where(temp_data)[1]-min_y]
    wei    = np.concatenate(smed[:,1], axis=0).reshape(3, len(bands), len(days), ysize, \
                            xsize)[:,:,16, np.where(temp_data)[0]-min_x, np.where(temp_data)[1]-min_y]
    std    = data.std(axis = 1)[:, :, np.where(temp_data)[0]-min_x, np.where(temp_data)[1]-min_y]
    print 'get angles...'
    va_files, sa_files = ang_files
    f      = lambda ang_file: reproject_data(ang_file, gdal.BuildVRT('', data_f[0]), outputType = gdal.GDT_Float64).data
    if isinstance(va_files[0], str):
        vas    = np.array(parmap(f, va_files))
    elif isinstance(sa_files[0], (np.ndarray, np.generic) ):
        f   = lambda array: reproject_data(array_to_raster(array, example_file), gdal.BuildVRT('', data_f[0])).data
        vas =  np.array(parmap(f, list(va_files)))
    vas = vas * sun_view_ang_scale[1]

    if isinstance(sa_files[0], str):
        sas = np.array(parmap(f, sa_files)) 
    elif isinstance(sa_files[0], (np.ndarray, np.generic) ):
        f   = lambda array: reproject_data(array_to_raster(array, example_file), gdal.BuildVRT('', data_f[0])).data
        sas =  np.array(parmap(f, list(sa_files)))
    sas     = np.repeat((sas * sun_view_ang_scale[0])[None, ...], len(bands), axis = 0)
    raa     = sas[:, 0, :, :] - vas[:, 0, :, :]
    angles  = vas[:, 1, temp_data], sas[:, 1, temp_data], raa[:, temp_data]
    kk      = get_kk(angles)
    k_vol   = kk.Ross
    k_geo   = kk.Li
    sur_ref = (dat[0] + dat[1]*k_geo + dat[2]*k_vol)*0.001
    wei    /= 0.05
    unc     = np.sqrt(std[:, 0, :]**2 + (std[:, 1, :]**2)*k_vol**2 + (std[:, 2, :]**2)*k_geo**2 + \
                      wei[0, :, :]**2 + (wei[1, :, :]**2)*k_vol**2 + (wei[2, :, :]**2)*k_geo**2) 
    unc     = np.minimum(unc*0.001, 0.5)
    f_dat   = np.repeat(temp_data[None, ...], len(bands), axis=0).astype(float)
    f_dat[:]= np.nan 
    unc_dat = f_dat.copy() 
    f_dat  [:, temp_data] = sur_ref
    unc_dat[:, temp_data] = unc
    f       = lambda array: reproject_data(array_to_raster(array, gdal.BuildVRT('', data_f[0])), example_file).data
    f_dat   = np.array(parmap(f, list(f_dat)))
    unc_dat = np.array(parmap(f, list(unc_dat))) 
    mask    = np.isnan(unc_dat) | (f_dat < 0.0001)
    f_dat   = np.ma.array(f_dat,   mask = mask)
    unc_dat = np.ma.array(unc_dat, mask = mask)
    return f_dat, unc_dat
