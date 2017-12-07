#/usr/bin/env python
import sys
from s2_aerosolNew import solve_aerosol
from s2_correction import atmospheric_correction
file_path = sys.argv[1]
s2_toa_dir = '/'.join(file_path.split('/')[:-8])
day        = int(file_path.split('/')[-3])
month      = int(file_path.split('/')[-4])
year       = int(file_path.split('/')[-5])
s2_tile = ''.join(file_path.split('/')[-8:-5])
aero = solve_aerosol(year, month, day, \
                     s2_toa_dir = s2_toa_dir,
                     mcd43_dir  = '/data/selene/ucfajlg/Hebei/MCD43/', \
                     emus_dir   = '/home/ucfafyi/DATA/Multiply/emus/', s2_tile=s2_tile, s2_psf=None)
aero.solving_s2_aerosol()
atm = atmospheric_correction(year, month, day, s2_tile, s2_toa_dir  = s2_toa_dir)
atm.atmospheric_correction()  
