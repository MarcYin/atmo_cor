for i in $(find ~/DATA/S2_MODIS/s_data/50/S/LG/ -path */0);do python Sentinel_atmo_cor.py $i/;done
