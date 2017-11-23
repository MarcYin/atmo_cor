import sys
sys.path.insert(0, 'python')
import gdal
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

def get_brdf_six(MCD43_dir, example_file, year, doy, angles, bands = (7,), Linds = None, do_unc = True):
    f_temp = MCD43_dir + '/MCD43A1.A%s.%s.006*.hdf'
    temp1  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'
    temp2  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'
    
    max_x, max_y = np.array(np.where(temp_data)).max(axis=1)
    min_x, min_y = np.array(np.where(temp_data)).min(axis=1)
    xoff,  yoff  = min_y, min_x
    xsize, ysize = (max_y - min_y + 1), (max_x - min_x + 1)
    
    unique_tile = get_hv(example_file)
    date   = datetime.strptime('%d%03d'%(year, doy), '%Y%j')
    days   = [(date - timedelta(days = i)).strftime('%Y%j') for i in np.arange(16, 0, -1)] + \
             [(date + timedelta(days = i)).strftime('%Y%j') for i in np.arange(0, 17,  1)]
    data_f = [[temp1%(glob.glob(f_temp%(day, tile))[0], band) for tile in unique_tile] for day in days for band in bands]  
    qa_f   = [[temp2%(glob.glob(f_temp%(day, tile))[0], band) for tile in unique_tile] for day in days for band in bands]

    driver = gdal.GetDriverByName('MEM')
    g = gdal.Open(example_file)
    ds = driver.Create('', 10980, 10980, 1, gdal.GDT_Byte)
    ds.SetProjection(g.GetProjection())
    ds.SetGeoTransform(g.GetGeoTransform())
    ds.GetRasterBand(1).WriteArray(np.ones((10980, 10980)))
    temp_data = reproject_data(ds, gdal.BuildVRT('', data_f[0])).data
    f = lambda fname: gdal.BuildVRT('', fname).ReadAsArray(xoff, yoff, xsize, ysize)
    data = np.array(parmap(f, data_f)).reshape(len(bands), len(days), 3, ysize, xsize)
    qa   = np.array(parmap(f, qa_f  )).reshape(len(bands), len(days),    ysize, xsize)
    w = 0.618034 ** qa.astype(float)
    f = lambda band: np.array(smoothn(data[band[0],:,band[1],:,:], s=2.5, smoothOrder=1, axis=0, TolZ=0.001, verbose=True, isrobust=True, W = w[band[0]])[0])
    ba = np.array([np.tile(range(len(bands)), 3), np.repeat(range(3), len(bands))]).T
    ret = parmap(f, ba)



    kk = get_kk(angles)
    k_vol = kk.Ross
    k_geo = kk.Li
    par = partial(r_modis, slic=Linds)
    ret = p.map(par, fnames)
    br, qa = np.array(ret[:len(bands)]), np.array(ret[len(bands):])
    if Linds is None:
        brdf = br[:,0,:,:] + br[:,1,:,:]*k_vol + br[:,2,:,:]*k_geo
    else:
        brdf = br[:,0] + br[:,1]*k_vol + br[:,2]*k_geo
    if do_unc:
        doy   = fname.split('.')[-5]
        date  = datetime.strptime(doy, 'A%Y%j')
        day_before = [(date - timedelta(days = i)).strftime('A%Y%j') for i in range(1,4)]
        day_after  = [(date + timedelta(days = i)).strftime('A%Y%j') for i in range(1,4)]
        finder = fname.split('MCD43A1')[0] + 'MCD43A1.%s.' + fname.split('.')[-4] +'.006.*hdf'
        before_f = sorted([glob(finder%i)[0] for i in day_before])
        after_f =  sorted([glob(finder%i)[0] for i in day_after])
        fnames = [temp1%(beforef, band) for beforef in before_f for band in bands] + \
                 [temp1%(afterf, band) for afterf in after_f for band in bands]

        p   = Pool(len(bands)*2)
        par = partial(r_modis, slic=Linds)
        ret = p.map(par, fnames)
        all_br = np.r_[np.array(ret).reshape((6,len(bands),3) + \
                       ret[0].shape[1:]), br.reshape((1,) + br.shape)]
        std = np.std(all_br, axis=0)
        if Linds is None:
            unc = np.sqrt(std[:,0,:,:]**2 + (std[:,1,:,:]**2)*k_vol**2 + (std[:,2,:,:]**2)*k_geo**2)

        else:
            unc = np.sqrt(std[:,0]**2 + (std[:,1]**2)*k_vol**2 + (std[:,2]**2)*k_geo**2)
        return [brdf*0.001, qa, unc*0.001]
    else:
        return [brdf*0.001, qa] 
