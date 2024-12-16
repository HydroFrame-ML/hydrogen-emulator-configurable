import pandas as pd
import os 
import numpy as np
import hf_hydrodata as hf
from parflow.tools.io import read_pfb, read_clm, write_pfb
import matplotlib.pyplot as plt
import netCDF4 as nc

#register PIN for hdyrodata
email = 'lecondon@email.arizona.edu'
pin = '1234'
print('Registering ' + email + ' (PIN=' + pin + ') for HydroData download' ) 
hf.register_api_pin(email, pin)

#directory with the root fracs and netcdfs for testing
working_dir = "./evap_trans_test" 
root_frac_dir = "/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/calculated_evaptrans"

#Set this to false if you don't want to regenerate the evaptrans pdfs
# i.e. if you are just doing comparison with the netcdf files for testing
run_calcs = True

#DZ list for CONUS2
dz_list = [0.1, 0.3, 0.6, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

#number of layers the root zone is distributed over in PF-CLM coupling
nroot_lay = 4 

#hour start and end for runn
hstart = 1
hend = 8760

#Water year for the CONUS simultions
wy= 2003

#CONUS2 dimensions
nz=10
ny=3256
nx=4442

# read in the root zone fractions
top_depth = np.append(0, np.cumsum(dz_list)) #Top depth of each layer
root_fracs=np.zeros((nroot_lay, ny, nx))
for layer in range(nroot_lay):
    #fin=os.path.join(working_dir, f'root_zone_frac_layer{layer}_{top_depth[layer]}-{top_depth[layer+1]}.pfb')
    fin=os.path.join(root_frac_dir, f'root_zone_frac_layer{layer}_{top_depth[layer]}-{top_depth[layer+1]}.pfb')
    root_fracs[layer,:,:] = read_pfb(fin)
    print(np.mean(root_fracs[layer,:,:]))

# Read in the mask file
options = {
      "dataset":"conus2_domain", "variable": "mask"}
mask = hf.get_gridded_data(options)
print(mask.shape)
print(np.sum(mask))

#Read in the top patch file
options = {
      "dataset":"conus2_domain", "variable": "top_patch"}
top_patch = hf.get_gridded_data(options)
print(top_patch.shape)

ltran = 8  #Layer of tran_veg in the clm_file
linfl = 9 #Layer of qflux_infl in the clm file

if run_calcs:
    print("Calculating Evap Trans from", hstart, "to", hend)
    for wy_hour in range(hstart, (hend+1)):
        print(wy_hour)

        #make an empty array for the evaptrans
        evap_trans = np.zeros((nz, ny, nx))

        #Read in the clm outputs
        fin1 = f"/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/run_inputs/spinup.wy{wy}.out.clm_output.{wy_hour:05d}.C.pfb"
        tran_veg = read_pfb(fin1, keys={'z': {'start': ltran, 'stop': (ltran+1),}})[0,::]
        infl = read_pfb(fin1, keys={'z': {'start': linfl, 'stop': (linfl+1),}})[0,:,:]

        # Calculate the evaptrans in the first layer
        # evap_trans[z] = qflx_tran_veg * rootfr[z] + qflx_infl + qflx_qirr_inst[z]
        # *3.6 because 1 mm/s = 3.6 m/hr
        # /dz to get to units of 1/hr
        evap_trans[0,:,:] = (-tran_veg * root_fracs[0,:,:] + infl) * mask * 3.6 /dz_list[0] # Assuming qirr=0

        #Calculate evaptrans in remaining clm layers
        # pf_flux[z]=(-qflx_tran_veg*rootfr[z]) + qflx_qirr_inst[z]
        for l in range(1,nroot_lay):
            evap_trans[l, :, :] = (- tran_veg * root_fracs[l,:,:]) * mask * 3.6 /dz_list[l] # Assuming qirr=0
        
        #Write it out as a pfb
        write_dir = "/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/calculated_evaptrans"
        fout = os.path.join(write_dir, f'calculated_evaptrans.{wy_hour:05d}.pfb')
        write_pfb(fout, evap_trans, dist=False, p=72, q=48, r=1)
    
    print(tran_veg.shape)
    print(np.sum(tran_veg[mask==1]), np.sum(infl[mask==1]))
else:
    print("Not recalculating the EvapTrans files")
