#!/usr/bin/python
import argparse
from l8_aerosol import solve_aerosol
from l8_correction import atmospheric_correction
parser = argparse.ArgumentParser(description='Landsat 8 Atmopsheric correction Excutable')
parser.add_argument('-p','--path', help='Landsat path number',required=True)
parser.add_argument('-r','--row', help='Landsat row number',required=True)
parser.add_argument('-d','--date',help='Sensing date in the format of: YYYYMMDD', required=True)
args = parser.parse_args()
year, month, day = int(args.date[:4]), int(args.date[4:6]), int(args.date[6:8])
aero = solve_aerosol(year, month, day, l8_tile = (int(args.path), int(args.row)), mcd43_dir   = '/data/nemesis/MCD43/')
aero.solving_l8_aerosol()
atmo_cor = atmospheric_correction(year, month, day, (int(args.path), int(args.row)))
atmo_cor.atmospheric_correction()
