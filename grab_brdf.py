import sys
sys.path.insert(0, 'python')
import gdal
import kernels
from osgeo import osr
from functools import partial
from multiprocessing import Pool
from reproject import reproject_data
from datetime import datetime, timedelta

x_step = -463.31271653
y_step = 463.31271653
m_y0, m_x0 = -20015109.354, 10007554.677

def mtile_cal(lat, lon):
    # a function calculate the tile number for MODIS, based on the lat and lon
    wgs84 = osr.SpatialReference( ) # Define a SpatialReference object
    wgs84.ImportFromEPSG( 4326 ) # And set it to WGS84 using the EPSG code
    modis_sinu = osr.SpatialReference() # define the SpatialReference object
    modis_sinu.ImportFromProj4 ( \
                    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
    tx = transform( wgs84, modis_sinu)# from wgs84 to modis 
    ho,vo,z = tx.TransformPoint(lon, lat)# still use the function instead of using the equation....
    h = int((ho-m_y0)/(2400*y_step))
    v = int((vo-m_x0)/(2400*x_step))
    return h,v


def get_hv(example_file):
    g = gdal.Open(H_res_fname)
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

def get_brdf_six(example_file, year, doy, root, angles, bands = (7,), Linds = None, do_unc = True):
    f_temp = root + 'MCD43A1.A%d%03d.%s.006*.hdf'
    temp1  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band%d'
    temp2  = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_Band%d'
    unique_tile = get_hv(example_file)
    data_g = [gdal.BuildVRT('', [gdal.Open(temp1%(glob.glob(f_temp%(year, doy, tile))[0], band)) for tile in unique_tile]) for band in bands]
    qa_g   = [gdal.BuildVRT('', [gdal.Open(temp2%(glob.glob(f_temp%(year, doy, tile))[0], band)) for tile in unique_tile]) for band in bands]
    max_x, max_y = np.array(np.where(reproject_data(example_file, data_g).data)).max(axis=1)
    min_x, min_y = np.array(np.where(reproject_data(example_file, data_g).data)).min(axis=1)
    


    p     = Pool(len(bands)*2)
    fnames = [temp1%(fname, band) for band in bands] + [temp2%(fname, band) for band in bands]
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
