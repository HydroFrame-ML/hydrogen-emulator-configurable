# Calculating mean and standard deviations of pressure deltas 

import matplotlib.pyplot as plt
import numpy as np
import os
from parflow import Run
from parflow.tools.io import read_pfb, read_clm, write_pfb
from parflow.tools.fs import mkdir
from parflow.tools.settings import set_working_directory
import subsettools as st
import hf_hydrodata as hf
import pandas as pd
import yaml

# Register the hydrodata pin
email = 'lecondon@email.arizona.edu' 
pin = '1234'
print('Registering ' + email + ' (PIN=' + pin + ') for HydroData download' ) #use lecondon@email.arizona.edu and 1234
hf.register_api_pin(email, pin)

# Read in the mask file
options = {
      "dataset":"conus2_domain", "variable": "mask"}
mask = hf.get_gridded_data(options)
print(mask.shape)
print(np.sum(mask))

# Set constants for reading
# interval is the interval that files will be read at (i.e. every interval hours)
# WY 2003 is the only transient year available for CONUS2 
# NZ is the number of layers in CONUS2

interval = 5 #picking a prime number here to ensure we don't grab the same time of day consistently
wy=2003
nz=10
hend= 8760 #hour to end at to do short test runs, set to 8760 to do the entire year

# Calculate the mean hourly pressure difference for every layer
# Loop through the year and get the delta pressures and calculate the mean
#Initialize some variables
wy_hour=interval + 5
pdif_sum = np.zeros(nz)
hour_count = 0

while wy_hour<=hend:
    print(wy_hour)
    
    fin1 = f"/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/run_inputs/spinup.wy{wy}.out.press.{wy_hour:05d}.pfb"
    fin0 = f"/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/run_inputs/spinup.wy{wy}.out.press.{(wy_hour-1):05d}.pfb"
    p1 = read_pfb(fin1)
    p0 = read_pfb(fin0)
    #print("read", fin1, " and ", fin0)
    pdif = p1 - p0

    for z in range(nz):
        pdif_z = pdif[z,:,:] 
        pdif_sum[z]=pdif_sum[z]+np.sum(pdif_z[mask==1])

    hour_count=hour_count + 1
    wy_hour=wy_hour+interval

pdif_mean = pdif_sum/(np.sum(mask)*hour_count)
#print(pdif_sum)
#print(pdif_mean)

# Calculate the standard deviation of hourly pressure differences
# Loop through the year and get the delta pressures and calculate the mean
#Initialize some variables
wy_hour=interval +5
numerator = np.zeros(nz)
hour_count = 0

while wy_hour<=hend:
    print(wy_hour)
    
    fin1 = f"/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/run_inputs/spinup.wy{wy}.out.press.{wy_hour:05d}.pfb"
    fin0 = f"/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/run_inputs/spinup.wy{wy}.out.press.{(wy_hour-1):05d}.pfb"
    p1 = read_pfb(fin1)
    p0 = read_pfb(fin0)
    pdif = (p1 - p0)

    #Q's: 1) Could I avoid creating pdif_z, 2)could I avoid the for loop
    #calculate a running sum of (pdif - pdif_mean)^2 for every layer
    for z in range(nz):
        pdif_mean_z = (pdif[z,:,:] - pdif_mean[z]) **2
        numerator[z]=numerator[z]+np.sum(pdif_mean_z[mask==1])

    hour_count=hour_count + 1
    wy_hour=wy_hour+interval

pdif_stdev = (numerator/(np.sum(mask)*hour_count))** 0.5
print(pdif_stdev)
print(pdif_mean)

# Save as CSV and YAML
fout = 'pressure_differnce_scalers_' + str(interval) + 'hour.csv'
row_names = ['layer_'+str(val) for val in range(nz)]
df=pd.DataFrame({'Name':row_names, 'Mean': pdif_mean, 'stdev': pdif_stdev})
df.set_index('Name')
df.to_csv(fout, index=False)

# Save as YAML
fout_yml = 'pressure_scalers_' + str(interval) + 'hour.yml'
with open(fout_yml, 'w') as file:
    yaml.dump(df.to_dict(orient='records'), file, sort_keys=False)
    
