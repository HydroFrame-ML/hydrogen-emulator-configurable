## CONUS 2 Data Prep

This folder contains scripts to prepare inputs needed for training on CONUS2. 

The `archive_notebooks` folder contains the original notebooks used to develop the workflows. Refer to these for more notes on methodology and for the cross checking that was done. 

1. Evaptrans was not output in the original simulations. So it was calculated after the fact here. 
    - First the root zone fractions were calculated from the landcover data `calculate_rootzone_fracs.ipynb` 
    - After root zones were calculated evaptrans was calculated using `calculate_evaptrans.py` this was run on verde `run_etcalc.slurm`

2. Scalers(mean and stdev) were calculated for all the static inputs, pressure difference between timesteps and evaptrans: 
   - Static Scalers: `calculate_static_scalers.ipynb`
   - Pressure differences: `pressure_scalers.py` + `run_pressurescalers.slurm`
   - Evaptrans: `evaptrans_scalers.py` + `run_evaptransscalers.slurm`
- **Note** that for the press and et scalers I just sampled every 5 hours to make the computations faster. 
- All calculations were done on verde
- Outputs are csv and yaml files. Output Yaml files were manually combined to create the `default_scalers.yaml` file

### Version note: 
The original version of this was done for CONUS2.0 WY2003 and results can be found here `/hydrodata/temp/CONUS2_transfers/CONUS2/spinup_WY2003/calculated_evaptrans`. To update for CONUS2.1 the root zone calculations were redone:
    1. using 5 root zone layers instead of 4 (the original 4 layer root zone fracs by landcover are in `root_zone_fractions_4layer.csv`)
    2. using the new landcover file that does not have any zeros in it
The outputs for the latest run are in `/hydrodata/temp/CONUS2.1/WY2003V2_run_outputs/calculated_evaptrans/`