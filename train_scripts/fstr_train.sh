#!/bin/bash
#
#SBATCH --job-name=fstrnn_train
#SBATCH --output=fstrnn_train.txt
#
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=200GB

TODAY=`date +"%Y-%m-%d"`
VARIANT="new_params_2l_64hd"
CONFIG=$(cat <<- EOM
{
    "resume_from_checkpoint": "/home/ab6361/hydrogen_workspace/notebooks_2023/forced_strnn_train_new_params_2l_64hd.pt",
    "train_dataset_files": [
        "/hydrodata/PFCLM/CONUS1_baseline/simulations/daily/zarr/conus1_2003_preprocessed.zarr",
        "/hydrodata/PFCLM/CONUS1_baseline/simulations/daily/zarr/conus1_2004_preprocessed.zarr"
    ],
    "valid_dataset_files": [
        "/hydrodata/PFCLM/CONUS1_baseline/simulations/daily/zarr/conus1_2005_preprocessed.zarr"
    ],
    "scaler_file": "/home/ab6361/hydrogen_workspace/data/new_scalers_may8.scalers",
    "log_dir": "/home/ab6361/hydrogen_workspace/artifacts/configurable_logs",
    "run_name": "forced_strnn_train_$VARIANT",
    "forcings": ["APCP", "Temp_max", "Temp_min", "melt", "et"],
    "parameters": [
        "topographic_index",
        "elevation",
        "frac_dist",
        "permeability_0",
        "permeability_1",
        "permeability_2",
        "porosity_0",
        "porosity_1",
        "porosity_2",
        "van_genuchten_alpha_0",
        "van_genuchten_alpha_1",
        "van_genuchten_alpha_2",
        "van_genuchten_n_0",
        "van_genuchten_n_1",
        "van_genuchten_n_2"
    ],
    "states": [
        "pressure_0",
        "pressure_1",
        "pressure_2",
        "pressure_3",
        "pressure_4"
    ],
    "targets": [
        "pressure_next_0",
        "pressure_next_1",
        "pressure_next_2",
        "pressure_next_3",
        "pressure_next_4"
    ],
    "sequence_length": 14,
    "patch_size": 48,
    "batch_size": 16,
    "num_workers": 8,
    "max_epochs": 1,
    "logging_frequency": 10,
    "model_def": {
        "type": "ForcedSTRNN",
        "config": {
            "num_layers": 2,
            "num_hidden": [64, 64],
            "img_channel": 5,
            "out_channel": 5,
            "act_channel": 5,
            "init_cond_channel": 5,
            "static_channel": 15
        }
    }
}
EOM
)
echo $CONFIG > train_config.json
run_emulator --mode train --domain subsurface --config train_config.json
