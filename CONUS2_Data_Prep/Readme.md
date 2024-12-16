## CONUS 2 Data Prep

This folder contains scripts to prepare inputs needed for training on CONUS2. 

The `archive_notebooks` folder contains the original notebooks used to develop the workflows. Refer to these for more notes on methodology and for the cross checking that was done. 

1. Evaptrans was not output in the original simulations. So it was calculated after the fact here. 
    - First the root zone fractions were calculated from the landcover data `calculate_rootzone_fracs.ipynb` (the resulting pfbs were put on hydrodata in the following folder `/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/calculated_evaptrans`)
    - **There is a disrepancy between the landcover pfb file for CONUS2 and the clm vegm file. Need to recreate the landcover file and re-run this. 
    - After root zones were calculated evaptrans was calculated using `calculate_evaptrans.py` this was run on verde `run_etcalc.slurm`

2. Scalers(mean and stdev) were calculated for all the static inputs, pressure difference between timesteps and evaptrans: 
   - Static Scalers: `calculate_static_scalers.ipynb`
   - Pressure differences: `pressure_scalers.py` + `run_pressurescalers.slurm`
   - Evaptrans: `evaptrans_scalers.py` + `run_evaptransscalers.slurm`
- **Note** that for the press and et scalers I just sampled every 5 hours to make the computations faster. 
- All calculations were done on verde
- Outputs are csv and yaml files. Output Yaml files were manually combined to create the overall scaler file. 